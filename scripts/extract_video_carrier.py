#!/usr/bin/env python3
"""
Extract a SynthID-style carrier pattern from the opening seconds of a video.

Method (single video, no reference set):
  1. Decode the first ``--seconds`` of video at native frame rate (ffmpeg).
  2. For each frame take luma Y and a high-pass noise residual
     (Y - Gaussian blur) to suppress scene content.
  3. Temporal consensus across frames:
       * mean residual     -> static additive pattern survives, moving content averages out
       * phase coherence   -> |mean(exp(i*phase))| per FFT bin; a fixed carrier gives ~1,
                              independent content gives ~1/sqrt(N)
  4. Carrier bins = coherence above a noise-floor threshold, outside the DC core.
     The carrier pattern is the inverse FFT of the mean residual restricted to those bins.
  5. Figures: carrier pattern, spectra, and the log-FFT spectrum overlaid on an example frame.

Usage::

    python scripts/extract_video_carrier.py video.mp4 --seconds 4 --out artifacts/video_carrier
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def read_frames(path: str, seconds: float) -> tuple[np.ndarray, float]:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,r_frame_rate", "-of", "json", path],
        capture_output=True, check=True, text=True,
    )
    st = json.loads(probe.stdout)["streams"][0]
    w, h = int(st["width"]), int(st["height"])
    num, den = st["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", path, "-t", str(seconds), "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        capture_output=True, check=True,
    ).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
    return frames, fps


def luma(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32)
    return 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]


def residual(y: np.ndarray, sigma: float) -> np.ndarray:
    return y - cv2.GaussianBlur(y, (0, 0), sigma)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--sigma", type=float, default=2.0, help="high-pass blur sigma")
    ap.add_argument("--dc-radius", type=int, default=8, help="bins to exclude around DC")
    ap.add_argument("--min-freq", type=float, default=0.05,
                    help="ignore bins below this radial freq (cycles/px); slow pans keep low-f content coherent")
    ap.add_argument("--nyquist-margin", type=float, default=0.03,
                    help="ignore bins within this distance of Nyquist (codec / scaler grid artefacts)")
    ap.add_argument("--example-frame", type=int, default=None)
    ap.add_argument("--out", default="artifacts/video_carrier")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    frames, fps = read_frames(args.video, args.seconds)
    n, H, W, _ = frames.shape
    print(f"{n} frames, {W}x{H} @ {fps:g} fps")

    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    res_sum = np.zeros((H, W), np.float64)
    unit_sum = np.zeros((H, W), np.complex128)
    mag_sum = np.zeros((H, W), np.float64)
    for f in frames:
        r = residual(luma(f), args.sigma)
        res_sum += r
        F = np.fft.fftshift(np.fft.fft2(r * win))
        mag = np.abs(F)
        mag_sum += mag
        unit_sum += F / np.maximum(mag, 1e-9)

    mean_res = res_sum / n
    mean_mag = mag_sum / n
    coherence = np.abs(unit_sum) / n
    mean_fft = np.fft.fftshift(np.fft.fft2(mean_res * win))

    # Noise floor for phase coherence with N independent random phases: E|.| ~ sqrt(pi/(4N)).
    floor = np.sqrt(np.pi / (4 * n))
    cy, cx = H // 2, W // 2
    yy, xx = np.mgrid[:H, :W]
    rr = np.hypot(yy - cy, xx - cx)
    fy, fx = (yy - cy) / H, (xx - cx) / W
    nyq = (np.abs(fx) > 0.5 - args.nyquist_margin) | (np.abs(fy) > 0.5 - args.nyquist_margin)
    valid = (rr > args.dc_radius) & (np.hypot(fx, fy) >= args.min_freq) & ~nyq
    bands = {}
    for lo, hi in [(0.0, 0.05), (0.05, 0.15), (0.15, 0.3), (0.3, 0.5)]:
        m = (rr > args.dc_radius) & (np.hypot(fx, fy) >= lo) & (np.hypot(fx, fy) < hi) & ~nyq
        bands[f"{lo}-{hi}"] = {"median_coherence": float(np.median(coherence[m])),
                               "frac_above_0.5": float(np.mean(coherence[m] > 0.5))}
    bands["nyquist_edges"] = {"median_coherence": float(np.median(coherence[nyq])),
                              "frac_above_0.5": float(np.mean(coherence[nyq] > 0.5))}
    for k, v in bands.items():
        print(f"  band {k:>14}: median coh {v['median_coherence']:.3f}, "
              f"frac>0.5 {v['frac_above_0.5']:.4f}")
    thr = max(5 * floor, float(np.percentile(coherence[valid], 99.9)))
    carrier_mask = (coherence >= thr) & valid
    print(f"coherence floor {floor:.3f}, threshold {thr:.3f}, "
          f"{int(carrier_mask.sum())} carrier bins")

    # Top carrier peaks (one per conjugate pair), frequencies in cycles/pixel.
    score = np.where(carrier_mask, coherence * np.log1p(np.abs(mean_fft)), 0)
    idx = np.argsort(score.ravel())[::-1]
    peaks, seen = [], set()
    for k in idx:
        if score.ravel()[k] <= 0 or len(peaks) >= 20:
            break
        y, x = divmod(int(k), W)
        key = tuple(sorted([(y - cy, x - cx), (cy - y, cx - x)]))
        if key in seen:
            continue
        seen.add(key)
        peaks.append({
            "fy_bin": y - cy, "fx_bin": x - cx,
            "fy_cpp": (y - cy) / H, "fx_cpp": (x - cx) / W,
            "period_px": float(1 / max(np.hypot((y - cy) / H, (x - cx) / W), 1e-9)),
            "coherence": float(coherence[y, x]),
            "magnitude": float(np.abs(mean_fft[y, x])),
        })

    carrier = np.real(np.fft.ifft2(np.fft.ifftshift(mean_fft * carrier_mask))) / np.maximum(win, 1e-3)
    carrier *= (win > 0.05)  # drop edges where window de-weighting blows up

    def norm(a, lo=1, hi=99):
        a0, a1 = np.percentile(a, [lo, hi])
        return np.clip((a - a0) / (a1 - a0 + 1e-12), 0, 1)

    np.savez_compressed(os.path.join(args.out, "carrier.npz"),
                        carrier=carrier.astype(np.float32),
                        mean_residual=mean_res.astype(np.float32),
                        coherence=coherence.astype(np.float32),
                        carrier_mask=carrier_mask, fps=fps, n_frames=n)
    with open(os.path.join(args.out, "carrier_peaks.json"), "w") as fh:
        json.dump({"n_frames": n, "fps": fps, "shape": [H, W],
                   "coherence_floor": floor, "threshold": thr,
                   "n_carrier_bins": int(carrier_mask.sum()),
                   "band_stats": bands, "peaks": peaks}, fh, indent=2)

    cv2.imwrite(os.path.join(args.out, "carrier_pattern.png"),
                (norm(carrier) * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.out, "mean_residual.png"),
                (norm(mean_res) * 255).astype(np.uint8))

    # --- summary figure ---------------------------------------------------
    fig, ax = plt.subplots(2, 3, figsize=(18, 10))
    ex = args.example_frame if args.example_frame is not None else n // 2
    ax[0, 0].imshow(frames[ex]); ax[0, 0].set_title(f"Example frame #{ex} (t={ex / fps:.1f}s)")
    ax[0, 1].imshow(norm(mean_res), cmap="gray"); ax[0, 1].set_title(f"Mean high-pass residual ({n} frames)")
    ax[0, 2].imshow(norm(carrier), cmap="RdBu_r"); ax[0, 2].set_title("Extracted carrier pattern (coherent bins only)")
    ax[1, 0].imshow(np.log1p(mean_mag), cmap="magma"); ax[1, 0].set_title("Mean |FFT| of residual (log)")
    ax[1, 1].imshow(coherence, cmap="viridis", vmin=0, vmax=1); ax[1, 1].set_title("Temporal phase coherence")
    ax[1, 2].imshow(np.log1p(np.abs(mean_fft)), cmap="magma")
    ys, xs = np.nonzero(carrier_mask)
    ax[1, 2].scatter(xs, ys, s=6, facecolors="none", edgecolors="cyan", linewidths=0.6)
    ax[1, 2].set_title(f"|FFT(mean residual)| + carrier bins ({len(xs)})")
    for a in ax.ravel():
        a.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "carrier_summary.png"), dpi=110)
    plt.close(fig)

    # --- FFT overlay on example frame -------------------------------------
    spec = norm(np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(luma(frames[ex]) * win)))), 50, 99.9)
    cmap = plt.get_cmap("inferno")(spec)[..., :3]
    base = frames[ex].astype(np.float32) / 255
    overlay = np.clip(0.35 * base + 1.2 * cmap * spec[..., None] ** 0.7, 0, 1)
    fig, ax = plt.subplots(1, 2, figsize=(20, 6.2))
    ax[0].imshow(overlay)
    ax[0].set_title(f"Frame #{ex} with its centred log-|FFT| spectrum overlaid")
    ax[1].imshow(overlay)
    ax[1].scatter(xs, ys, s=10, facecolors="none", edgecolors="cyan", linewidths=0.7)
    ax[1].set_title("Same overlay + temporally coherent carrier bins (cyan)")
    for a in ax:
        a.set_xticks([cx - W // 4, cx, cx + W // 4]); a.set_xticklabels(["-0.25", "0", "+0.25"])
        a.set_yticks([cy - H // 4, cy, cy + H // 4]); a.set_yticklabels(["-0.25", "0", "+0.25"])
        a.set_xlabel("fx (cycles/px)"); a.set_ylabel("fy (cycles/px)")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "fft_overlay.png"), dpi=110)
    plt.close(fig)

    for p in peaks[:10]:
        print(f"  f=({p['fx_cpp']:+.4f},{p['fy_cpp']:+.4f}) cpp  period {p['period_px']:.1f}px  "
              f"coh {p['coherence']:.3f}")


if __name__ == "__main__":
    main()
