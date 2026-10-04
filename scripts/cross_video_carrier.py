#!/usr/bin/env python3
"""
Cross-video SynthID carrier consensus.

A true carrier is content-independent, so its FFT phase must be (a) stable over time
within each video and (b) identical across unrelated videos at the same resolution.
Codec artefacts also satisfy (a) and (b) when they sit on the macroblock grid, so bins
within +-1 of a k/16 cycles-per-pixel harmonic are excluded.

Per video: luma high-pass residual -> per-bin temporal phase coherence |mean(e^{i phi})|.
Across videos: carrier bins = coherent (> --tau) in every video, off the codec grid,
and phase-aligned (min pairwise cos(dphi) > --min-cos). The carrier pattern is the inverse
FFT over those bins with consensus phase and mean residual amplitude.

Usage::

    python scripts/cross_video_carrier.py a.mp4 b.mp4 [c.mp4 ...] --out artifacts/video_carrier_consensus
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def probe(path: str) -> tuple[int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", path],
        capture_output=True, check=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    return int(st["height"]), int(st["width"])


def video_stats(path: str, H: int, W: int, sigma: float, seconds: float | None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    raw = subprocess.run(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    win = np.outer(np.hanning(H), np.hanning(W))
    unit = np.zeros((H, W), np.complex128)
    fsum = np.zeros((H, W), np.complex128)
    for f in frames:
        f = f.astype(np.float32)
        y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        F = np.fft.fft2((y - cv2.GaussianBlur(y, (0, 0), sigma)) * win)
        unit += F / np.maximum(np.abs(F), 1e-9)
        fsum += F
    n = len(frames)
    return unit / n, fsum / n, frames[n // 2], n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--tau", type=float, default=0.5, help="per-video coherence threshold")
    ap.add_argument("--min-cos", type=float, default=0.8, help="min pairwise cos(dphase)")
    ap.add_argument("--min-freq", type=float, default=0.05)
    ap.add_argument("--overlay-video", type=int, default=-1, help="index of video for overlay frame")
    ap.add_argument("--out", default="artifacts/video_carrier_consensus")
    args = ap.parse_args()
    if len(args.videos) < 2:
        ap.error("need at least two videos")
    os.makedirs(args.out, exist_ok=True)

    shapes = {probe(v) for v in args.videos}
    if len(shapes) != 1:
        raise SystemExit(f"videos must share a resolution, got {shapes}")
    H, W = shapes.pop()

    stats = [video_stats(v, H, W, args.sigma, args.seconds) for v in args.videos]
    units = [s[0] for s in stats]
    cohs = [np.abs(u) for u in units]
    for v, s, c in zip(args.videos, stats, cohs):
        print(f"{os.path.basename(v)}: {s[3]} frames, coherence floor {np.sqrt(np.pi / (4 * s[3])):.3f}")

    ky = np.round(np.fft.fftfreq(H) * H).astype(int)[:, None]
    kx = np.round(np.fft.fftfreq(W) * W).astype(int)[None, :]
    rad = np.hypot(ky / H, kx / W)

    def near_grid(k, N):  # within +-1 bin of a k/16 cpp harmonic
        step = N / 16
        return np.abs(k - np.round(k / step) * step) <= 1

    grid = near_grid(ky, H) & near_grid(kx, W)
    coherent = np.logical_and.reduce([c > args.tau for c in cohs]) & (rad >= args.min_freq) & ~grid
    pair_cos = [np.cos(np.angle(a * np.conj(b))) for a, b in itertools.combinations(units, 2)]
    min_cos = np.minimum.reduce(pair_cos)
    carrier = coherent & (min_cos > args.min_cos)
    carrier &= carrier[(-np.arange(H)) % H][:, (-np.arange(W)) % W]  # keep conjugate pairs

    # Chance expectation of joint coherence
    p = [np.mean((c > args.tau) & (rad >= args.min_freq) & ~grid) for c in cohs]
    print(f"coherent in all videos: {int(coherent.sum())} bins (chance ~{np.prod(p) * H * W:.1f}); "
          f"phase-aligned carrier bins: {int(carrier.sum())}")

    cons_unit = np.mean(units, axis=0)
    amp = np.mean([np.abs(s[1]) for s in stats], axis=0)
    spec = np.where(carrier, amp * np.exp(1j * np.angle(cons_unit)), 0)
    win = np.outer(np.hanning(H), np.hanning(W))
    pattern = np.real(np.fft.ifft2(spec)) / np.maximum(win, 1e-3) * (win > 0.05)

    peaks = []
    for y, x in np.argwhere(carrier):
        if ky[y, 0] < 0 or (ky[y, 0] == 0 and kx[0, x] < 0):
            continue  # one per conjugate pair
        peaks.append({"ky": int(ky[y, 0]), "kx": int(kx[0, x]),
                      "fy_cpp": ky[y, 0] / H, "fx_cpp": kx[0, x] / W,
                      "coherence": [float(c[y, x]) for c in cohs],
                      "min_pair_cos": float(min_cos[y, x]),
                      "phase": float(np.angle(cons_unit[y, x]))})
    peaks.sort(key=lambda d: -min(d["coherence"]))
    with open(os.path.join(args.out, "consensus_carriers.json"), "w") as fh:
        json.dump({"videos": [os.path.basename(v) for v in args.videos], "shape": [H, W],
                   "tau": args.tau, "min_cos": args.min_cos,
                   "n_coherent_all": int(coherent.sum()), "chance_coherent_all": float(np.prod(p) * H * W),
                   "carriers": peaks}, fh, indent=2)
    np.savez_compressed(os.path.join(args.out, "consensus_carrier.npz"),
                        carrier_mask=carrier, consensus_phase=np.angle(cons_unit).astype(np.float32),
                        pattern=pattern.astype(np.float32))
    for d in peaks[:15]:
        print(f"  (ky,kx)=({d['ky']:>4},{d['kx']:>5})  f=({d['fy_cpp']:.4f},{d['fx_cpp']:+.4f}) cpp  "
              f"period {1 / np.hypot(d['fy_cpp'], d['fx_cpp']):.2f}px  coh {np.round(d['coherence'], 2).tolist()}  "
              f"cos {d['min_pair_cos']:+.2f}")

    def norm(a, lo=1, hi=99):
        a0, a1 = np.percentile(a, [lo, hi])
        return np.clip((a - a0) / (a1 - a0 + 1e-12), 0, 1)

    cv2.imwrite(os.path.join(args.out, "consensus_carrier_pattern.png"), (norm(pattern) * 255).astype(np.uint8))

    # Figure: per-video coherence, pattern, overlay on example frame
    sh = np.fft.fftshift
    k = len(args.videos)
    fig, ax = plt.subplots(2, max(k, 2), figsize=(7 * max(k, 2), 9.5))
    for i, (v, c) in enumerate(zip(args.videos, cohs)):
        ax[0, i].imshow(sh(c), cmap="viridis", vmin=0, vmax=1)
        ax[0, i].set_title(f"Temporal phase coherence\n{os.path.basename(v)}")
    ys, xs = np.nonzero(sh(carrier))
    frame = stats[args.overlay_video][2]
    lspec = norm(np.log1p(np.abs(sh(np.fft.fft2(
        (0.299 * frame[..., 0] + 0.587 * frame[..., 1] + 0.114 * frame[..., 2]) * win)))), 50, 99.9)
    over = np.clip(0.35 * frame / 255 + 1.2 * plt.get_cmap("inferno")(lspec)[..., :3] * lspec[..., None] ** 0.7, 0, 1)
    ax[1, 0].imshow(over)
    ax[1, 0].scatter(xs, ys, s=40, facecolors="none", edgecolors="cyan", linewidths=1.2)
    ax[1, 0].set_title(f"log|FFT| over frame ({os.path.basename(args.videos[args.overlay_video])})\n"
                       f"cyan = cross-video carrier bins ({len(xs)})")
    ax[1, 1].imshow(norm(pattern), cmap="RdBu_r")
    ax[1, 1].set_title("Consensus carrier pattern (spatial)")
    for i in range(2, max(k, 2)):
        ax[1, i].axis("off")
    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "consensus_summary.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
