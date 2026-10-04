#!/usr/bin/env python3
"""
Score videos against a saved video-carrier key (``consensus_carriers.json``).

For each key bin (ky, kx) with key phase phi_k, the video's per-bin statistic is
    s = |mean_t e^{i phi_t}| * cos(angle(mean_t e^{i phi_t}) - phi_k)
i.e. temporal phase coherence signed by agreement with the key. The video score is the
mean of s over key bins. The null distribution draws the same number of random off-grid
bins from the same frequency band with random key phases (``--null-draws`` times), giving
a z-score that controls for how static / coherent the video is overall.

Usage::

    python scripts/score_video_carrier.py --key artifacts/video_carrier_consensus/consensus_carriers.json v1.mp4 v2.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np


def mean_unit_phasor(path: str, H: int, W: int, sigma: float, seconds: float | None) -> np.ndarray:
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    raw = subprocess.run(cmd + ["-vf", f"scale={W}:{H}:flags=neighbor", "-f", "rawvideo",
                                "-pix_fmt", "rgb24", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    win = np.outer(np.hanning(H), np.hanning(W))
    u = np.zeros((H, W), np.complex128)
    for f in frames:
        f = f.astype(np.float32)
        y = 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]
        F = np.fft.fft2((y - cv2.GaussianBlur(y, (0, 0), sigma)) * win)
        u += F / np.maximum(np.abs(F), 1e-9)
    return u / len(frames)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--key", required=True)
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--null-draws", type=int, default=2000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    key = json.load(open(args.key))
    H, W = key["shape"]
    bins = [(d["ky"] % H, d["kx"] % W, d["phase"]) for d in key["carriers"]]
    kys = np.array([d["ky"] for d in key["carriers"]])
    kxs = np.array([d["kx"] for d in key["carriers"]])
    rng = np.random.default_rng(0)

    ky_all = np.round(np.fft.fftfreq(H) * H).astype(int)
    kx_all = np.round(np.fft.fftfreq(W) * W).astype(int)

    def near_grid(k, N):
        step = N / 16
        return np.abs(k - np.round(k / step) * step) <= 1

    # Null pool: same ky range and |kx| range as the key, off the macroblock grid, not key bins.
    pool_y = np.nonzero((ky_all >= kys.min()) & (ky_all <= kys.max()))[0]
    pool_x = np.nonzero(np.abs(kx_all) <= max(np.abs(kxs).max(), 1) + 6)[0]
    PY, PX = np.meshgrid(pool_y, pool_x, indexing="ij")
    ok = ~(near_grid(ky_all[PY], H) & near_grid(kx_all[PX], W))
    keyset = {(y, x) for y, x, _ in bins}
    ok &= np.array([[(y, x) not in keyset for x in pool_x] for y in pool_y])
    PY, PX = PY[ok], PX[ok]

    results = []
    for v in args.videos:
        u = mean_unit_phasor(v, H, W, args.sigma, args.seconds)
        s = np.array([np.abs(u[y, x]) * np.cos(np.angle(u[y, x]) - p) for y, x, p in bins])
        n = len(bins)
        idx = rng.integers(0, len(PY), (args.null_draws, n))
        null = (np.abs(u[PY[idx], PX[idx]]) * np.cos(rng.uniform(-np.pi, np.pi, idx.shape))).mean(1)
        score = float(s.mean())
        z = (score - null.mean()) / (null.std() + 1e-12)
        p = float((np.sum(null >= score) + 1) / (args.null_draws + 1))
        mean_cos = float(np.mean([np.cos(np.angle(u[y, x]) - ph) for y, x, ph in bins]))
        r = {"video": os.path.basename(v), "score": score, "z": float(z), "p_emp": p,
             "mean_cos": mean_cos, "mean_coh_key": float(np.mean([np.abs(u[y, x]) for y, x, _ in bins])),
             "median_coh_pool": float(np.median(np.abs(u[PY, PX])))}
        results.append(r)
        print(f"{r['video'][:48]:48s} score {score:+.3f}  z {z:+6.2f}  p {p:.4f}  "
              f"mean cos {mean_cos:+.2f}  coh key {r['mean_coh_key']:.2f} (pool median {r['median_coh_pool']:.2f})")
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"key": args.key, "results": results}, fh, indent=2)


if __name__ == "__main__":
    main()
