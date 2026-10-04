# Google SynthID Detector results (ground truth from the user, 2026-10-04)

Images were checked by the user with **Google's SynthID Detector**. Test-pack images were produced in
this session; the micrograph frames are lossless PNG grabs from `1st-2026_Xu_Ning_e42800_f27376.mp4`.

## Positive pack — detector: SynthID present in every image
The detector's summary: "Analysis confirms the presence of a SynthID watermark in every single one of
them, indicating that most or all of each image was generated or edited using Google AI."

| image | what it is |
|---|---|
| `01_original_frame6_unedited.png` | frame 6 of the original video, no edits |
| `05_original_frame010/050/100/150` | original frames at t = 1, 5, 10, 15 s, no edits |
| `02_frame6_regraded_like_still_no_still_pixels.png` | frame 6 with the still's colour grade / sharpening fitted by regression |
| `03_blend_50pct_still_50pct_frame6regraded.png` | 50/50 blend of the still and #02 |
| `04_still_resaved_png.png` | the user's still (`1.jpg`) re-saved as PNG |
| `06_POSITIVE_CONTROL_veo_remake_frame.png` | frame of the Veo 3.1 remake |

## Negative pack — detector: "No reliable signals were detected"
| image | what it is |
|---|---|
| `NEG_real_microscopy_IHC_1920x1072.png` | scikit-image `immunohistochemistry` (real micrograph), upscaled |
| `NEG_real_photo_coffee_1920x1072.png` | scikit-image `coffee` (real photo), upscaled |
| `NEG_real_photo_cat_1920x1072.png` | scikit-image `chelsea` (real photo), upscaled |
| `NEG_synthetic_mandelbrot_x264.png` | FFmpeg Mandelbrot frame, libx264 |
| `NEG_synthetic_testpattern_x264.png` | FFmpeg testsrc2 frame, libx264 |

Caveat: verdicts were reported as one summary per pack, not per image.

## Conclusion
The detector separates the packs cleanly, and it flags the **unedited** original frames across the
video's length. The SynthID watermark is therefore present in the original micrograph video itself;
the still inherited it from frame 6 and was not watermarked by its later colour edit.
By Google's detector, the original video was generated or edited with Google AI.

## What this means for the analyses in this repo
* `artifacts/veo_watermark_model/` — the 720p Veo/Gemini carrier family remains valid as a description
  of the fixed pattern shared by the 1280x720 Google videos.
* `artifacts/video_carrier/`, `artifacts/veo_watermark_detection/` — the null results for the original
  video are **false negatives** of the pattern-based detector: it only recognises the fixed 720p carrier,
  and the original is a 1920x1072, 10 fps FFmpeg re-encode. They must not be read as evidence that the
  original is real footage.
* `artifacts/still_paired_analysis/` — frame 6 was assumed un-watermarked; it is watermarked, so the
  paired subtraction cancels the watermark instead of isolating it. The null there is expected.
