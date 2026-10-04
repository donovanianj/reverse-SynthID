#!/usr/bin/env python3
"""
Build a model of the fixed luma pattern shared by Google Veo / Gemini 1280x720 videos.

Discovery (see artifacts/veo_watermark_model/NOTES.md) showed the strongest phase-locked,
content-independent bins form one family, with ky in cycles per frame height and kx in cycles
per frame width (FFT of a 1280x720 frame):

    row components     (14 m, 0)    m in {7, 9, 11, 15, 17}
                       ((14*13, 0) and (14*16, 0) also phase-lock across videos but lie within
                        2 bins of H.264 macroblock-row harmonics 180 = 4*45 and 225 = 5*45, and
                        (224, 0) is strongly coherent in a stock-x264 control, so both are excluded)
    tilted components  (14 m, +14)  m in {9, 13, 17}        (m = 1 mod 4)
                       (-14 m, +14) m in {7, 11, 15}        (m = 3 mod 4)

This script measures that family on each input video and writes:
  * model.json: bins, frequencies, consensus phase, amplitude per video (peak luma levels),
    and leave-one-out phase validation
  * template.png / template.npy: the synthesised spatial pattern (consensus phase, median amplitude)
  * model_summary.png: discovery map, template, per-frame agreement over time

Usage::

    python scripts/veo_watermark_model.py white.mp4 cilia.mp4 veo.mp4 --out artifacts/veo_watermark_model
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

H, W = 720, 1280
FAMILY = ([(14 * m, 0) for m in (7, 9, 11, 15, 17)]
          + [(14 * m, 14) for m in (9, 13, 17)]
          + [(-14 * m, 14) for m in (7, 11, 15)])
SIGMA = 2.0


def frames(path: str):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    for f in fr:
        f = f.astype(np.float32)
        yield 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


def hp_gain(f: float) -> float:
    """Gain of (1 - Gaussian blur) at radial frequency f (cycles/px)."""
    return 1 - np.exp(-2 * np.pi ** 2 * SIGMA ** 2 * f ** 2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+", help="1280x720 Veo/Gemini videos")
    ap.add_argument("--out", default="artifacts/veo_watermark_model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
    wsum = float(win.sum())
    ys = np.array([b[0] % H for b in FAMILY])
    xs = np.array([b[1] for b in FAMILY])
    names = [os.path.basename(v) for v in args.videos]

    unit, cmean, per_frame_F, unit_full = {}, {}, {}, {}
    for v, n in zip(args.videos, names):
        U = np.zeros((H, W // 2 + 1), np.complex128)
        Fb = []
        for y in frames(v):
            F = np.fft.rfft2((y - cv2.GaussianBlur(y, (0, 0), SIGMA)) * win)
            U += F / np.maximum(np.abs(F), 1e-9)
            Fb.append(F[ys, xs])
        Fb = np.array(Fb)
        unit_full[n] = U / len(Fb)
        unit[n] = unit_full[n][ys, xs]
        cmean[n] = Fb.mean(0)
        per_frame_F[n] = Fb
        print(f"{n}: {len(Fb)} frames")

    # Discovery map: mean pairwise phase agreement Re(U_a conj U_b) over all pairs
    pairs = list(itertools.combinations(names, 2))
    q = sum(np.real(unit_full[a] * np.conj(unit_full[b])) for a, b in pairs) / len(pairs)

    consensus = np.angle(sum(unit[n] for n in names))
    loo = {}
    for held in names:
        ref = np.angle(sum(unit[n] for n in names if n != held))
        c = np.cos(np.angle(unit[held]) - ref)
        loo[held] = {"mean_cos": float(c.mean()), "n_positive": int((c > 0).sum()), "n": len(c),
                     "mean_coherence": float(np.abs(unit[held]).mean())}
        print(f"  LOO held {held}: mean cos {c.mean():+.2f}, {int((c > 0).sum())}/{len(c)} positive")

    bins = []
    for i, (ky, kx) in enumerate(FAMILY):
        f = float(np.hypot(ky / H, kx / W))
        amps = {n: float(2 * np.abs(cmean[n][i]) / wsum / hp_gain(f)) for n in names}
        bins.append({"ky": ky, "kx": kx, "fy_cpp": ky / H, "fx_cpp": kx / W, "f_cpp": f,
                     "period_px": 1 / f, "consensus_phase": float(consensus[i]),
                     "amplitude_luma": amps, "amplitude_median": float(np.median(list(amps.values()))),
                     "coherence": {n: float(np.abs(unit[n][i])) for n in names}})

    # Spatial template (full frame, luma levels)
    yy, xx = np.mgrid[:H, :W]
    tmpl = np.zeros((H, W), np.float64)
    for b in bins:
        tmpl += b["amplitude_median"] * np.cos(2 * np.pi * (b["ky"] * yy / H + b["kx"] * xx / W)
                                               + b["consensus_phase"])
    np.save(os.path.join(args.out, "template.npy"), tmpl.astype(np.float32))
    t8 = np.clip((tmpl - tmpl.min()) / (np.ptp(tmpl) + 1e-12) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(args.out, "template.png"), t8)

    model = {"shape": [H, W], "sigma_highpass": SIGMA, "videos": names,
             "family_rule": "ky=14m (m in 7,9,11,15,17) kx=0; (14m,+14) m=1 mod 4; (-14m,+14) m=3 mod 4",
             "template_peak_to_peak_luma": float(np.ptp(tmpl)),
             "template_rms_luma": float(tmpl.std()),
             "leave_one_out": loo, "bins": bins}
    with open(os.path.join(args.out, "model.json"), "w") as fh:
        json.dump(model, fh, indent=2)
    print(f"template rms {tmpl.std():.3f} luma levels, peak-to-peak {np.ptp(tmpl):.3f}")

    # Figure
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    qs = np.fft.fftshift(q, axes=0)[:, :80]
    ax.imshow(qs[H // 2 - 300:H // 2 + 300], cmap="RdBu_r", vmin=-0.4, vmax=0.4, aspect="auto",
              extent=[0, 80, 300, -300])
    ax.scatter([b["kx"] for b in bins], [b["ky"] for b in bins], s=80, facecolors="none",
               edgecolors="k", linewidths=1.2)
    ax.set_xlabel("kx (cycles / frame width)"); ax.set_ylabel("ky (cycles / frame height)")
    ax.set_title("Cross-video phase agreement Re(U_a U_b*)\n(circles = carrier family)")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(tmpl, cmap="RdBu_r"); ax.set_title(f"Synthesised template (rms {tmpl.std():.3f} luma)")
    ax.axis("off")
    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(tmpl[300:404, 560:744], cmap="RdBu_r", interpolation="nearest")
    ax.set_title("Template, 104x184 px crop"); ax.axis("off")
    ax = fig.add_subplot(gs[1, :2])
    for n in names:
        s = np.cos(np.angle(per_frame_F[n]) - consensus).mean(1)
        ax.plot(np.arange(len(s)) / 24, s, lw=1, label=f"{n[:40]} (mean {s.mean():+.2f})")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("time (s)"); ax.set_ylabel("per-frame mean cos(phase - consensus)")
    ax.set_title("Per-frame agreement with the carrier phase"); ax.legend(fontsize=8)
    ax = fig.add_subplot(gs[1, 2])
    lab = [f"({b['ky']},{b['kx']})" for b in bins]
    for n in names:
        ax.plot([b["amplitude_luma"][n] for b in bins], "o-", ms=4, label=n[:24])
    ax.set_xticks(range(len(bins))); ax.set_xticklabels(lab, rotation=70, fontsize=7)
    ax.set_ylabel("amplitude (peak luma levels)"); ax.set_title("Per-bin amplitude"); ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "model_summary.png"), dpi=100)
    plt.close(fig)


if __name__ == "__main__":
    main()
