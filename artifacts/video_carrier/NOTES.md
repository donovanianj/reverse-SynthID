# 1st-2026_Xu_Ning_e42800_f27376.mp4 — fluorescence micrograph video

> **Correction (2026-10-04):** Google's SynthID Detector reports SynthID in unedited frames of the
> original micrograph video and none in negative controls; see `artifacts/SYNTHID_DETECTOR_RESULTS.md`.
> The null results below are false negatives of the 720p-carrier method, not evidence the video is real footage.

1920x1072, 10 fps, 19.3 s, H.264 ~6.8 Mb/s, `encoder=Lavf57.25.100` (FFmpeg, not Google).
(Uploaded twice; both copies are byte-identical, sha256 8dcdf401...)

## Single-video extraction (first 4 s, `scripts/extract_video_carrier.py`)
No fixed carrier. Coherence above 0.15 cycles/px sits at the noise floor (0.134 vs 0.14);
low-frequency "carrier" bins are slowly panning content (~30 px / 4 s); remaining peaks are
H.264 8-px block-grid harmonics (fy = 2/8, 3/8) and Nyquist-column artefacts.

## Against the 1280x720 Veo/Gemini key (all 193 frames)
Key: `artifacts/video_carrier_consensus/consensus_carriers.json` (held-out Veo 3.1 z = +5.5).

| Mapping of key bins | coherence key / pool | z | p | phase |
|---|---|---|---|---|
| native 1920x1072, same cycles/px | 0.069 / 0.058 | +0.9 | 0.14 | n/a |
| native 1920x1072, same cycles/frame | 0.085 / 0.059 | +2.2 | 0.024 | n/a |
| area-downscaled to 1280x720, key phases | 0.086 / 0.059 | +2.2 | 0.025 | mean cos -0.33 |

Noise floor for 193 frames is 0.064. The cycles/frame result is weak, not significant after
correcting for the three mappings tried (~0.07), and the one phase-aware test disagrees with
the key phase. **No evidence of the Veo/Gemini 720p carrier.** This does not show the video is
not AI-generated: rescaling to 1920x1072 and FFmpeg re-encoding at a different resolution would
displace or destroy a pixel-grid carrier, and the key has not been validated at other resolutions.
