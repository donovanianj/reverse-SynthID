# Cross-video carrier consensus — two Gemini videos (1280x720, 24 fps, 10 s)

| Video | Content | Bitrate |
|---|---|---|
| `gemini_generated_video_498A9F35.mp4` | solid white | ~0.75 Mb/s |
| `gemini_generated_video_2C5D4F75.mp4` | fluorescence micrograph, ciliated epithelium | ~3.4 Mb/s |

Command: `python scripts/cross_video_carrier.py <white.mp4> <cilia.mp4> --out artifacts/video_carrier_consensus`

## Result: a shared, phase-locked, off-codec-grid set of bins

- Luma-residual bins with temporal phase coherence > 0.5 in **both** videos, at least 0.05 cycles/px
  and not within +-1 bin of a k/16 cycles/px macroblock harmonic: **20** (chance ~0.4).
- Of those, **18** (9 conjugate pairs) are phase-aligned across videos (cos dphi > 0.8; most 0.98–1.00).
- All lie at **ky in {126, 182, 210, 238} cycles/frame height** (14 x 9, 13, 15, 17; 125 is a leakage neighbour)
  and **kx in {0, +-14} cycles/frame width**.

| (ky, kx) | f (cycles/px) | period | coherence white / cilia | cos dphi |
|---|---|---|---|---|
| (126, 14) | (0.175, 0.011) | 5.7 px | 1.00 / 0.77 | +1.00 |
| (210, -14) | (0.292, -0.011) | 3.4 px | 0.99 / 0.67 | +0.99 |
| (126, 0) | (0.175, 0) | 5.7 px | 0.99 / 0.64 | +0.89 |
| (182, 14) | (0.253, 0.011) | 4.0 px | 0.98 / 0.61 | +0.98 |
| (238, 14) | (0.331, 0.011) | 3.0 px | 0.98 / 0.59 | +0.99 |
| (210, 0) | (0.292, 0) | 3.4 px | 0.98 / 0.56 | +1.00 |

Spatially (`consensus_carrier_pattern.png`): fine horizontal striping (3–6 px vertical period)
whose amplitude is modulated by 14 bands across the width (~91 px). Luma only.

Supporting test (exploratory, `kx = 0` and `+-14` rows): odd multiples of 14/H agree across videos
(mean cos +0.54 / +0.47); even multiples do not (+0.07 / +0.05). A scan of the fundamental (12–16/H)
is not uniquely peaked at 14, so the "odd harmonics of 14" description is suggestive, not established.

## Caveats

- Only two videos. Consistent with a SynthID-style fixed key, but equally consistent with any
  deterministic stage shared by Gemini's video path (upscaler, VAE tiling, Google's encoder settings).
  Needed controls: a **non-Gemini 1280x720 video**, ideally passed through the same encoder path, and
  more Gemini videos (different content / colours) to test that the bin set and phases stay fixed.
- The coherent column-stripe line along fy = 0 in the cilia video is **not** shared with the white
  video, so it is not part of the consensus.
- The earlier 1920x1072 micrograph video (`artifacts/video_carrier/`) is a different resolution and
  cannot be compared bin-for-bin.
