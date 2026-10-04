# Paired analysis of the still (`1.jpg`, 1918x1072 JPEG) against its source frame

Exploratory scripts in this folder (paths point at a session scratchpad; adjust before reuse).

## What the still is
* Frame 6 (t = 0.6 s) of `1st-2026_Xu_Ning_e42800_f27376.mp4`, cropped 1 px left/right.
  Phase correlation: shift (0.05, -0.05) px, scale 1.000, response 0.96. No regeneration:
  nuclear texture and membranes match pixel for pixel (`crop_frame6_vs_still.png`).
* Edits: per-channel colour grade (gains B 1.04, G 0.92, R 1.14), sharpening/contrast, and
  JPEG with stronger 8x8 blocking than a q90 re-save.

## Content-free residual
Each still channel regressed on a nonlinear, multi-scale feature set of the aligned frame
(colours, products, cubes, Laplacians at 0.7–6 px). Unexplained residual rms (B/G/R):
still 6.1 / 4.0 / 7.2 vs JPEG-only control 2.2 / 1.4 / 2.1. The excess follows edges and detail
(`residual_energy_map.png`) — the sharpening/grade, not an additive pattern. In flat dark
background, residual rms is 0.74 vs 0.66 for the control: at most ~0.33 levels of extra signal.

## Watermark searches (all null)
| search | still | paired control |
|---|---|---|
| sparse periodic peaks in residual spectrum | only JPEG 8-px grid harmonics (k/8 cpp) | — |
| background-patch spectra (B/G/R) | no peaks | no peaks |
| Veo video key, single image | best z +1.8 (27 offsets, Bonf. p 0.91) | frame 6: +1.8 |
| Veo video key on content-free residual | best z +1.5 | JPEG-ctrl residual +0.8 |
| repo V4 Gemini-image codebook, 14 profiles x 3 ch, raw still | max z +7.7 (gemini 843x1264, B) | frame 6 +5.6, frame6-JPEG +5.7 on the same profile: content-driven |
| repo V4 codebook on content-free residual | all within +-3.9 over 42 tests | JPEG-ctrl residual: within +-3.9 |
| repo legacy detector | not watermarked (phase match 0.504) | frame 6: 0.522 |

The analytic phase-match null is not valid for natural images (low-frequency content phases are
not uniform), which is why the raw-still V4 numbers must be read against frame 6.

## Conclusion
No SynthID carrier recognisable by any model available here (Veo/Gemini 720p video carrier;
Gemini-3.1-flash-image and nano-banana-pro image codebooks) is present in the still beyond what its
own un-watermarked source frame shows. If the still carries SynthID, it is a variant not covered
by these models, and its pixel-domain footprint in flat regions is <= ~0.3 luma levels rms.
