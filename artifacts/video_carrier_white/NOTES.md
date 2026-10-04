# White-background Gemini video — calibration run

Source: `gemini_generated_video_498A9F35.mp4` — 1280x720, 24 fps, 10 s, H.264 ~750 kb/s,
`encoder=Google`. Mean luma ~250, frame-to-frame |diff| ~0.1.

Command: `python scripts/extract_video_carrier.py <video> --seconds 4 --out artifacts/video_carrier_white`

## Result: no isolable SynthID carrier in the decoded pixels

| Test | Value | Reading |
|---|---|---|
| Temporal phase coherence, 0.15–0.3 cpp (96 frames) | median 0.23 vs floor 0.09 | something static is present |
| Same, I-frames only (26 independently coded frames) | median 0.27 vs floor 0.17 | survives independent intra coding |
| Mean residual, I vs B frames (0.1–0.35 cpp band) | r = 0.81 | |
| First 4 s vs last 4 s | r = 0.61 | pattern drifts |
| Frame 0 vs frame 234 (both I-frames) | r = 0.06 | not a fixed pixel-grid carrier |
| Chroma residual / luma residual | ~0.14 | luma-only |
| 8x8 luma blocks that are exactly constant | 50% (38–71% per frame) | sub-LSB dither erased |

The "carrier pattern" (horizontal banding, top peaks at fy = 0.175, 0.136, 0.125 cpp)
is H.264 structure: the residual exists only along the 8-bit quantisation contours of the
white vignette and is made of horizontally elongated 4/8/16-px partitions (see
`crop_frame0_stretched.png`, `crop_mean_Iframe_residual.png`). Flat plateaus carry no
residual at all, so a low-amplitude additive watermark has nowhere to survive there.
The slow drift follows the global brightness creep (250.0 -> 249.8), which moves the contours.

Implication: unlike Gemini still images (where a white background isolates the carrier),
this delivery path re-encodes the video hard enough that pixel-domain spectral averaging
cannot recover a SynthID carrier. Either the video watermark is not a fixed spatial
carrier, or it is embedded in content/texture regions that a flat white video lacks.
