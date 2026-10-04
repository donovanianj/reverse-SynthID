# 1080p cross-video consensus: original micrograph vs Veo 3.1 remake

Both are confirmed SynthID-positive by Google's SynthID Detector (`artifacts/SYNTHID_DETECTOR_RESULTS.md`).
Exploratory scripts here (`u1080.py`, `x1080.py`; paths point at a session scratchpad).

Method: per video, the temporal mean unit phasor of the luma high-pass FFT on a 1920x1080 canvas
(the 1920x1072 original zero-padded with dy = 0..8 rows on top). Cross-video phase agreement
cos(phi_remake - phi_original) on bins coherent in both, 0.08–0.45 cycles/px, macroblock-grid
harmonics excluded; null = same with the original's map shifted by (9, 13) bins.

## Results: no shared 1080p carrier

| dy | bins coherent in both | mean cos | frac > 0 | shifted-null mean cos |
|---|---|---|---|---|
| 0–8 | 1919–1936 | -0.020 to +0.015 | 0.485–0.509 | -0.029 to +0.023 |

* The original has essentially **no temporally fixed pattern** in the mid/high band: median coherence
  0.060 vs the 193-frame noise floor 0.064. Only 261 off-grid bins >= 0.05 cpp exceed 0.35:
  160 near Nyquist (codec/scaler), 20 near fx ~ 1/3 (peak at kx = 642, period 2.99 px — not an exact
  3:2 resampling harmonic; exact-1/3 coherence is 0.27 vs 0.95 in a bicubic x1.5 control), 36 below
  0.15 cpp (slowly panning content).
* The remake is coherent at 135 of those bins but with opposite mean phase (cos -0.17): shared
  codec/resampling frequencies, not a shared key.
* Side finding: Veo's native-1080p output has a strong period-3 row pattern (coherence 0.94 at
  fy = 1/3), consistent with internal vertical 720 -> 1080 upscaling.

## Interpretation
Google's detector finds SynthID in every tested frame of the original, yet the original contains no
fixed-phase spatial carrier that temporal averaging can recover, and nothing in common with the remake
at 1080p. So the watermark in the original is not a static pixel-grid pattern after its processing
(FFmpeg re-encode, 10 fps, 1920x1072): it is content-adaptive and/or varies frame to frame, or is
too weak after re-encoding for phase averaging. This is the limit of the pattern-based approach used
in this repo; recovering it would need Google's decoder or a learned detector trained on labelled data.
