#!/usr/bin/env python3
"""
Single-image test for the Veo/Gemini 720p carrier family (``model.json``).

A still has no temporal averaging, so the statistic is the per-frame one:
    s = mean_b cos(angle F_b - phi_b)        (11 carrier bins)
For a random image the phases are uniform, so s has mean 0 and sd sqrt(0.5 / n); the script
also reports an empirical null from random structure-matched frequency sets.

Geometry: the image is padded (edge replicate) back to 16:9 over a grid of (dx, dy) offsets,
area-resized to 1280x720, and scored; the best offset is Bonferroni-corrected.

Usage::

    python scripts/detect_veo_watermark_still.py img1.png img2.jpg --model artifacts/veo_watermark_model/model.json
"""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from detect_veo_watermark import DFT, null_freqs


def candidates(img: np.ndarray, W0: int = 1280, H0: int = 720):
    """Yield (label, 1280x720 luma) for each pad offset that restores a 16:9 frame."""
    H, W = img.shape
    Wt = int(np.ceil(max(W, H * 16 / 9) / 16) * 16)   # 1918 -> 1920, 1920x1072 -> 1920
    Ht = int(round(Wt * 9 / 16))
    padx, pady = Wt - W, Ht - H
    for dx in range(padx + 1):
        for dy in range(pady + 1):
            y = cv2.copyMakeBorder(img, dy, pady - dy, dx, padx - dx, cv2.BORDER_REPLICATE)
            yield f"dx={dx},dy={dy}", cv2.resize(y, (W0, H0), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--model", default="artifacts/veo_watermark_model/model.json")
    ap.add_argument("--null-sets", type=int, default=2000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    model = json.load(open(args.model))
    H0, W0 = model["shape"]
    fy = np.array([b["fy_cpp"] for b in model["bins"]])
    fx = np.array([b["fx_cpp"] for b in model["bins"]])
    ph = np.array([b["consensus_phase"] for b in model["bins"]])
    n = len(ph)
    rng = np.random.default_rng(0)
    nf = null_freqs(fy, fx, 300, rng, H0, W0)
    dft = DFT(H0, W0, np.concatenate([fy, nf[:, 0]]), np.concatenate([fx, nf[:, 1]]))

    out = []
    for p in args.images:
        bgr = cv2.imread(p).astype(np.float32)
        y = 0.299 * bgr[..., 2] + 0.587 * bgr[..., 1] + 0.114 * bgr[..., 0]
        rows = []
        for lab, y720 in candidates(y, W0, H0):
            F = dft(y720)
            s = float(np.mean(np.cos(np.angle(F[:n]) - ph)))
            Fn = F[n:]
            idx = rng.integers(0, len(Fn), (args.null_sets, n))
            nul = np.cos(np.angle(Fn[idx]) - rng.uniform(-np.pi, np.pi, idx.shape)).mean(1)
            rows.append((s, lab, float((s - nul.mean()) / nul.std())))
        rows.sort(reverse=True)
        s, lab, z = rows[0]
        k = len(rows)
        z_an = s / np.sqrt(0.5 / n)
        from math import erfc, sqrt
        p1 = 0.5 * erfc(z_an / sqrt(2))
        r = {"image": os.path.basename(p), "shape": list(y.shape), "best_offset": lab, "score": s,
             "z_empirical": z, "z_analytic": float(z_an), "p_one_offset": p1,
             "p_bonferroni": min(1.0, p1 * k), "n_offsets": k,
             "median_score_over_offsets": float(np.median([r_[0] for r_ in rows]))}
        out.append(r)
        print(f"{r['image'][:30]:30s} {y.shape[1]}x{y.shape[0]}  best {lab:10s} s {s:+.3f}  z {z_an:+.2f}  "
              f"p_bonf {r['p_bonferroni']:.3f} ({k} offsets; median s {r['median_score_over_offsets']:+.3f})")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"model": args.model, "results": out}, fh, indent=2)


if __name__ == "__main__":
    main()
