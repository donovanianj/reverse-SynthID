#!/usr/bin/env python3
"""
Build a probe set from one SynthID-positive image for localisation with Google's SynthID Detector
used as a black-box oracle (upload each probe individually and record the verdict).

The probes answer *where* the watermark lives (spatial regions, luma vs chroma, coarse
low/high frequency split). Strength ladders (JPEG quality, noise, blur sweeps until the detector
flips) are intentionally not generated: they measure removal thresholds, not structure.

Usage::

    python scripts/make_oracle_probes.py still.jpg --negative neg.png --out probes/
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image")
    ap.add_argument("--negative", required=True, help="known SynthID-negative image (control)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--regions", default=None,
                    help="optional 'name:x,y,w,h;name:x,y,w,h' content crops (pixels)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    im = cv2.imread(args.image)
    H, W = im.shape[:2]
    f = im.astype(np.float32)
    probes = {}

    # Controls
    probes["E1_control_full_image"] = im
    probes["E2_control_known_negative"] = cv2.imread(args.negative)

    # A: spatial halves (native pixels, no rescaling)
    probes["A1_top_half"] = im[: H // 2]
    probes["A2_bottom_half"] = im[H // 2:]
    probes["A3_left_half"] = im[:, : W // 2]
    probes["A4_right_half"] = im[:, W // 2:]
    if args.regions:
        for i, spec in enumerate(args.regions.split(";")):
            name, box = spec.split(":")
            x, y, w, h = map(int, box.split(","))
            probes[f"A{5 + i}_region_{name}"] = im[y:y + h, x:x + w]

    # D: colour — luma only, and chroma only (luma flattened to its mean)
    ycc = cv2.cvtColor(im, cv2.COLOR_BGR2YCrCb)
    probes["D1_luma_only_grayscale"] = cv2.cvtColor(ycc[..., 0], cv2.COLOR_GRAY2BGR)
    flat = ycc.copy()
    flat[..., 0] = int(ycc[..., 0].mean())
    probes["D2_chroma_only_flat_luma"] = cv2.cvtColor(flat, cv2.COLOR_YCrCb2BGR)

    # C: coarse frequency split (complementary: C1 + C2 - mean = original)
    low = cv2.GaussianBlur(f, (0, 0), 3.0)
    probes["C1_low_frequencies_only"] = np.clip(low, 0, 255).astype(np.uint8)
    probes["C2_high_frequencies_only_on_mean"] = np.clip(f - low + f.mean((0, 1)), 0, 255).astype(np.uint8)

    for name, p in probes.items():
        cv2.imwrite(os.path.join(args.out, f"{name}.png"), p)
        print(f"{name}.png  {p.shape[1]}x{p.shape[0]}")


if __name__ == "__main__":
    main()
