# Detection: original micrograph video, full length

> **Correction (2026-10-04):** Google's SynthID Detector reports SynthID in unedited frames of the
> original micrograph video and none in negative controls; see `artifacts/SYNTHID_DETECTOR_RESULTS.md`.
> The original-video "no trace" conclusion below is a false negative of the pattern-based detector, which only recognises the fixed 720p carrier.

`scripts/detect_veo_watermark.py`. Statistic S = mean over the 11 carrier bins of
|temporal mean phasor| x cos(phase - key phase); null = structure-matched random frequencies
(same fx values, random fy in band, off the macroblock grid) with random key phases.
For non-720p input, each crop offset dy (pad back to 16:9, area-resize to 1280x720) is tested and
the best is Bonferroni-corrected; a phase-free coherence test needs no geometry.

## Controls (model built WITHOUT the video being tested)

| video | pipeline | best z | p | frames > 0 |
|---|---|---|---|---|
| Veo 3.1 (held out) | original | +5.8 | 0.0002 | 99% |
| Veo 3.1 | 10 fps, libx264 3 Mb/s, 720p | +4.8 | 0.0002 | 96% |
| Veo 3.1 | **micrograph pipeline**: 1.5x up, crop to 1920x1072 (dy=4), 10 fps, x264 6.8 Mb/s | +4.9 at dy=4 | 0.002 (Bonf.) | 95% |
| Gemini cilia (held out, weaker watermark) | micrograph pipeline | +11.2 at dy=4 | 0.002 (Bonf.) | 90% |
| Mandelbrot zoom, x264 | — | +0.7 | 0.23 | 51% |
| panning blurred-noise texture, x264 | — | +0.6 | 0.26 | 53% |
| white vignette + noise, x264 | — | +0.5 | 0.31 | 52% |
| testsrc2 test pattern, x264 | — | -3.2 | 1.00 | 7% |

The geometry search recovers the true crop offset (dy = 4) and adjacent offsets fall off smoothly;
(testsrc2 is a synthetic pattern with strong deterministic structure, which gives a large negative,
not a false positive, under the one-sided test.)

## Original video: `1st-2026_Xu_Ning_e42800_f27376.mp4`

1920x1072, 10 fps, all **193 frames** (19.3 s), FFmpeg-encoded (Lavf57).

| test | full model | Gemini-only model |
|---|---|---|
| phase-free (cycles/frame) | z +0.42, p 0.30 | z +0.42, p 0.30 |
| best resample offset | dy=8: z +1.40, Bonferroni p 0.70 | dy=2: z +1.35, Bonferroni p 0.78 |
| per-frame mean / frames > 0 | +0.02 / ~55% | +0.02 / ~55% |

All nine offsets: |z| < 1.7, per-frame agreement 42–58% (chance = 50%).

**Conclusion: no trace of the Veo/Gemini 720p carrier anywhere in the original video**, under
geometries where the same carrier, put through the same processing, is detected at z = 4.9 (Veo)
and z = 11.2 (Gemini). Not covered: a source generated natively at 1080p (carrier unknown at that
resolution), heavier processing (e.g. denoising, AI upscaling, re-generation), or non-Google generators.

## Matched-content control: Veo 3.1 remake of the micrograph (1920x1080, 24 fps, 8 s, 12 Mb/s)

`Generated_Video_October_04_2026_-_12_26PM.mp4`, generated with the original as template; not used
in the model.

| input | test | result |
|---|---|---|
| remake, native 1080p, downscaled to 720p | phase-aware | S +0.17, **z +2.9, p 0.002**, 79% frames > 0 |
| remake, native 1080p | phase-free, cycles/frame | z +0.8 (static video: high background coherence) |
| remake through original's pipeline (crop to 1072, 10 fps, x264 6.8 Mb/s) | best offset | **dy=4 (true): z +2.8, Bonferroni p 0.018**, 81% frames > 0 (80 frames) |
| original (193 frames) | best offset | dy=8: z +1.4, Bonferroni p 0.70, ~55% frames > 0 |

Scaled to the original's 193 frames, a watermarked equivalent would be expected near z ~4.4.
The 720p carrier is present but ~2x weaker in the native 1080p Veo output than in 720p outputs; a
native-1080p carrier key cannot be learned from a single 1080p video (needs >= 2 for consensus).

## Single still image (`1.jpg`, 1918x1072 JPEG, no EXIF)

The still is frame 6 (t = 0.6 s) of the original video, shifted 1 px horizontally
(r = 0.985, mean |diff| 5 levels: a separate grab/encode of the same frame).

`scripts/detect_veo_watermark_still.py` (single-frame score, 27 pad offsets, Bonferroni):

| image | best z | Bonferroni p |
|---|---|---|
| **user still** | +1.8 (s +0.39) | **0.91** |
| original frames (t = 0.6, 1, 3, 5, 7 s) | +0.5 to +2.3 | 0.09–1.0 |
| Veo 3.1 remake frames (1080p) | +0.9 to +2.2 | 0.01–0.19 |
| Veo 3.1 goldfish frames (720p) | +0.5 to +2.9 | 0.002–0.30 |
| x264 noise-texture frames | -1.5 to -0.2 | > 0.5 |

A single frame carries too little of this sub-LSB carrier for a reliable decision: known-Veo frames
are often not significant individually. The still is therefore inconclusive on its own; since it is
a frame of the original video, the full-video result (no carrier) is the stronger evidence.

The repo's still-image SynthID detectors (`robust_extractor.detect_array`, Gemini *image* codebooks)
return "not watermarked" for every frame tested, including known-Veo frames, so they are not
informative for Veo video frames (different watermark / resolution profiles).
