#!/usr/bin/env python3
"""
Detect the Veo/Gemini 1280x720 carrier family (``model.json`` from veo_watermark_model.py)
in a video of any resolution, frame by frame and over the whole video.

Fourier coefficients are evaluated at exact (possibly fractional) frequencies with a direct
DFT, so no resampling of the spectrum is needed.

Hypotheses tested
-----------------
* ``native-720p``: input is 1280x720; key frequencies and phases used directly.
* ``resample dy=k``: input was produced by scaling 1280x720 up to a 16:9 frame and then
  cropping rows (e.g. 1920x1080 -> 1920x1072). The frame is padded back to 16:9 with ``k`` rows
  on top (edge replicate), area-resized to 1280x720, and scored with phases. ``k`` is searched;
  the best z is Bonferroni-corrected.
* ``phase-free``: temporal phase coherence at the key frequencies mapped to the native frame in
  cycles per frame (cropping ignored), vs. random nearby frequencies. Does not need geometry.

Video statistic (phase-aware): S = mean_b |U_b| cos(angle U_b - phi_b), with U_b the temporal
mean unit phasor. Null: the same statistic on random off-family frequencies from the same band
with random reference phases. Per-frame statistic: mean_b cos(angle F_b,t - phi_b).

Usage::

    python scripts/detect_veo_watermark.py video.mp4 [more.mp4] --model artifacts/veo_watermark_model/model.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess

import cv2
import numpy as np

SIGMA = 2.0


def probe(path: str) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", path],
                         capture_output=True, check=True, text=True).stdout
    st = json.loads(out)["streams"][0]
    a, b = st["r_frame_rate"].split("/")
    return int(st["height"]), int(st["width"]), float(a) / float(b)


def luma_frames(path: str, H: int, W: int, seconds: float | None):
    cmd = ["ffmpeg", "-v", "error", "-i", path]
    if seconds:
        cmd += ["-t", str(seconds)]
    raw = subprocess.run(cmd + ["-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         capture_output=True, check=True).stdout
    fr = np.frombuffer(raw, np.uint8).reshape(-1, H, W, 3)
    for f in fr:
        f = f.astype(np.float32)
        yield 0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2]


class DFT:
    """Windowed Fourier coefficients of the high-passed frame at arbitrary (fy, fx) cycles/px."""

    def __init__(self, H: int, W: int, fy: np.ndarray, fx: np.ndarray):
        self.win = np.outer(np.hanning(H), np.hanning(W)).astype(np.float32)
        self.Ey = np.exp(-2j * np.pi * np.outer(fy, np.arange(H))).astype(np.complex64)   # (n, H)
        self.Ex = np.exp(-2j * np.pi * np.outer(fx, np.arange(W))).astype(np.complex64)   # (n, W)

    def __call__(self, y: np.ndarray) -> np.ndarray:
        r = ((y - cv2.GaussianBlur(y, (0, 0), SIGMA)) * self.win).astype(np.float32)
        return np.einsum("nh,hw,nw->n", self.Ey, r.astype(np.complex64), self.Ex, optimize=True)


def null_freqs(fy_key, fx_key, n, rng, H, W):
    """Random frequencies matched to the key's structure: each null frequency reuses one of the
    key's exact fx values (row-only bins stay row-only) with a random fy from the key's |fy| band,
    >= 3 bins from every key bin and off the k/16 macroblock grid."""
    lo, hi = np.abs(fy_key).min(), np.abs(fy_key).max()
    fx_choices = np.unique(np.round(fx_key * W)) / W
    out = []
    while len(out) < n:
        fy = rng.uniform(lo, hi) * rng.choice([-1, 1])
        fx = rng.choice(fx_choices)
        if np.min(np.hypot((fy_key - fy) * H, (fx_key - fx) * W)) < 3:
            continue
        if np.min(np.hypot((-fy_key - fy) * H, (-fx_key - fx) * W)) < 3:   # conjugate key bins
            continue
        if abs(fy * 16 - round(fy * 16)) * H / 16 < 1.5 and abs(fx * 16 - round(fx * 16)) * W / 16 < 1.5:
            continue
        out.append((fy, fx))
    return np.array(out)


def score(unit_key, unit_null, phases, rng, draws=4000):
    s = float(np.mean(np.abs(unit_key) * np.cos(np.angle(unit_key) - phases)))
    n = len(phases)
    idx = rng.integers(0, len(unit_null), (draws, n))
    u = unit_null[idx]
    nul = (np.abs(u) * np.cos(np.angle(u) - rng.uniform(-np.pi, np.pi, idx.shape))).mean(1)
    z = (s - nul.mean()) / (nul.std() + 1e-12)
    p = (np.sum(nul >= s) + 1) / (draws + 1)
    return s, float(z), float(p)


def run(frames_iter, H, W, fy, fx, phases, rng, n_null=400):
    nf = null_freqs(fy, fx, n_null, rng, H, W)
    dft = DFT(H, W, np.concatenate([fy, nf[:, 0]]), np.concatenate([fx, nf[:, 1]]))
    nk = len(fy)
    U = 0
    per_frame = []
    T = 0
    for y in frames_iter:
        F = dft(y)
        u = F / np.maximum(np.abs(F), 1e-9)
        U = U + u
        per_frame.append(float(np.mean(np.cos(np.angle(u[:nk]) - phases))) if phases is not None else np.nan)
        T += 1
    U = U / T
    return U[:nk], U[nk:], np.array(per_frame), T


def pad_resize(frames_iter, H, W, dy, out_h=720, out_w=1280):
    H169 = int(round(W * 9 / 16))
    pad_total = H169 - H
    for y in frames_iter:
        if pad_total > 0:
            y = cv2.copyMakeBorder(y, dy, pad_total - dy, 0, 0, cv2.BORDER_REPLICATE)
        elif pad_total < 0:
            y = y[dy:dy + H169]
        yield cv2.resize(y, (out_w, out_h), interpolation=cv2.INTER_AREA)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--model", default="artifacts/veo_watermark_model/model.json")
    ap.add_argument("--seconds", type=float, default=None)
    ap.add_argument("--max-dy", type=int, default=None, help="max crop offset searched (default: all)")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--plot", default=None, help="save per-frame plot (png)")
    args = ap.parse_args()

    model = json.load(open(args.model))
    MH, MW = model["shape"]
    kfy = np.array([b["fy_cpp"] for b in model["bins"]])
    kfx = np.array([b["fx_cpp"] for b in model["bins"]])
    kph = np.array([b["consensus_phase"] for b in model["bins"]])
    kky = np.array([b["ky"] for b in model["bins"]])
    kkx = np.array([b["kx"] for b in model["bins"]])
    rng = np.random.default_rng(0)
    results, series = [], {}

    for v in args.videos:
        H, W, fps = probe(v)
        name = os.path.basename(v)
        print(f"\n{name}: {W}x{H} @ {fps:g} fps")
        res = {"video": name, "shape": [H, W], "fps": fps, "tests": []}

        if (H, W) == (MH, MW):
            uk, un, pf, T = run(luma_frames(v, H, W, args.seconds), H, W, kfy, kfx, kph, rng)
            s, z, p = score(uk, un, kph, rng)
            res["tests"].append({"hypothesis": "native-720p", "S": s, "z": z, "p": p, "frames": T,
                                 "per_frame_mean": float(pf.mean()), "per_frame_frac_pos": float(np.mean(pf > 0))})
            series[f"{name} native"] = (pf, fps)
            print(f"  native-720p: S {s:+.3f} z {z:+.2f} p {p:.4f} | per-frame mean {pf.mean():+.3f}, "
                  f"{np.mean(pf > 0) * 100:.0f}% frames > 0 ({T} frames)")
        else:
            # Phase-free: key frequencies in cycles/frame mapped to native frame
            fy_n, fx_n = kky / H, kkx / W
            uk, un, _, T = run(luma_frames(v, H, W, args.seconds), H, W, fy_n, fx_n, None, rng)
            ck, cn = np.abs(uk), np.abs(un)
            idx = rng.integers(0, len(cn), (4000, len(ck)))
            nul = cn[idx].mean(1)
            zf = float((ck.mean() - nul.mean()) / nul.std())
            pfree = float((np.sum(nul >= ck.mean()) + 1) / 4001)
            res["tests"].append({"hypothesis": "phase-free cycles/frame", "coh_key": float(ck.mean()),
                                 "coh_null": float(nul.mean()), "z": zf, "p": pfree, "frames": T})
            print(f"  phase-free (cycles/frame): coherence key {ck.mean():.3f} vs null {nul.mean():.3f}  "
                  f"z {zf:+.2f} p {pfree:.4f} ({T} frames)")
            # Geometry search
            H169 = int(round(W * 9 / 16))
            pad = abs(H169 - H)
            dys = range(0, (pad if args.max_dy is None else min(pad, args.max_dy)) + 1)
            best = None
            for dy in dys:
                uk, un, pf, T = run(pad_resize(luma_frames(v, H, W, args.seconds), H, W, dy),
                                    MH, MW, kfy, kfx, kph, rng)
                s, z, p = score(uk, un, kph, rng)
                t = {"hypothesis": f"resample dy={dy}", "S": s, "z": z, "p": p, "frames": T,
                     "per_frame_mean": float(pf.mean()), "per_frame_frac_pos": float(np.mean(pf > 0))}
                res["tests"].append(t)
                print(f"  resample dy={dy}: S {s:+.3f} z {z:+.2f} p {p:.4f} | per-frame mean "
                      f"{pf.mean():+.3f}, {np.mean(pf > 0) * 100:.0f}% frames > 0")
                if best is None or z > best[0]["z"]:
                    best = (t, pf)
            n_h = len(list(dys))
            res["best_resample"] = {**best[0], "p_bonferroni": min(1.0, best[0]["p"] * n_h), "n_hypotheses": n_h}
            series[f"{name} {best[0]['hypothesis']}"] = (best[1], fps)
            print(f"  best resample {best[0]['hypothesis']}: z {best[0]['z']:+.2f}, "
                  f"Bonferroni p {min(1.0, best[0]['p'] * n_h):.4f} over {n_h} offsets")
        results.append(res)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"model": args.model, "results": results}, fh, indent=2)
    if args.plot and series:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4.5))
        for lab, (pf, fps) in series.items():
            ax.plot(np.arange(len(pf)) / fps, pf, lw=1, label=f"{lab[:60]} (mean {pf.mean():+.2f})")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_xlabel("time (s)"); ax.set_ylabel("per-frame mean cos(phase - key)")
        ax.set_title("Per-frame Veo carrier agreement (0 = chance)"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(args.plot, dpi=100); plt.close(fig)


if __name__ == "__main__":
    main()
