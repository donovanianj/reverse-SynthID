# Reverse-engineered Veo / Gemini video carrier (1280x720)

Built from three Google videos (`scripts/veo_watermark_model.py`):
Gemini white (0.75 Mb/s), Gemini cilia micrograph (3.4 Mb/s), Veo 3.1 goldfish (3.1 Mb/s, known SynthID).

## What the pattern is

A fixed, additive, luma-only pattern, identical in phase across videos and static over time.
In FFT bins of a 1280x720 frame (ky = cycles per frame height, kx = cycles per frame width):

| component | bins | note |
|---|---|---|
| row stripes | (14m, 0), m = 7, 9, 11, 15, 17 | periods 7.3, 5.7, 4.7, 3.4, 3.0 px |
| tilted, m = 1 mod 4 | (14m, +14), m = 9, 13, 17 | |
| tilted, m = 3 mod 4 | (-14m, +14), m = 7, 11, 15 | opposite-sign partners are absent |

So every component has vertical frequency on the 14-cycles-per-height lattice (fundamental period
720/14 = 51.4 rows, odd harmonics 7-17), and the tilted ones share one horizontal frequency
(14 cycles per width, 91.4 px), with the tilt direction alternating by m mod 4.
(14*13, 0) and (14*16, 0) also phase-lock but sit within 2 bins of H.264 macroblock-row harmonics
(180, 225) and (224, 0) is strongly coherent in a stock-x264 control, so they are excluded.

* Amplitude: 0.002–0.06 peak luma levels per component; synthesised template rms **0.035** levels
  (peak-to-peak 0.25), i.e. far below one 8-bit step. It survives only statistically (dither/averaging).
* Relative amplitude profile is the same in all three videos; Veo 3.1 is ~2.5x stronger overall.
  Strongest components: (98, 0), (126, +14), (-98, +14).
* Temporal: phase is constant through each clip; per-frame agreement > 0 in 100% / 96% / 99% of frames.

## Validation

Leave-one-out (bins fixed, phases learned from the other two videos):

| held out | positive bins | mean cos dphi |
|---|---|---|
| Gemini white | 11/11 | +0.96 |
| Gemini cilia | 11/11 | +0.91 |
| Veo 3.1 | 10/11 | +0.75 |

Detector controls are in `artifacts/veo_watermark_detection/NOTES.md`.

## Limits
* 1280x720 only. Other resolutions (1080p, portrait) have not been observed and may use other bins.
* The data cannot separate SynthID from another fixed Google-side stage; it is consistent with
  SynthID (present in a known-SynthID Veo 3.1 clip, absent from all non-Google controls).
