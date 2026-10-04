# Held-out validation on a Veo 3.1 video (known SynthID)

Video: `Generated_Video_October_03_2026_-_10_38PM.mp4` — Veo 3.1, 1280x720, 24 fps, 8 s,
~3.1 Mb/s, `encoder=Google`. Content: goldfish bowl, ink, lotus (moving).

## 1. Held-out score against the 2-video key

The key (`artifacts/video_carrier_consensus/consensus_carriers.json`, 9 conjugate pairs at
ky in {125,126,182,210,238}, kx in {0,13,+-14}) was built from the two Gemini videos only.

Command:
`python scripts/score_video_carrier.py --key artifacts/video_carrier_consensus/consensus_carriers.json <videos...>`

Score = mean over key bins of coherence x cos(phase - key phase); null = 2000 draws of random
off-grid bins from the same band with random key phases.

| Video | Role | score | z | p (empirical) | mean cos dphi |
|---|---|---|---|---|---|
| Veo 3.1 goldfish | **held out** | +0.42 | **+5.5** | <= 0.0005 | +0.81 |
| micrograph (non-Google) -> 720p, libx264 3 Mb/s | control | +0.00 | +0.1 | 0.49 | +0.04 |
| synthetic white vignette + noise, libx264 750 kb/s | control | -0.03 | -1.2 | 0.88 | -0.42 |
| Gemini white | training | +0.98 | +11.3 | <= 0.0005 | +1.00 |
| Gemini cilia | training | +0.60 | +20.2 | <= 0.0005 | +0.99 |

All 9 key pairs have positive phase agreement in the Veo video (cos +0.58 to +0.99).
Training-video rows are not independent evidence; they are shown for scale.

## 2. Three-video consensus (`consensus_carriers.json` here)

Bins coherent (> 0.5) in all three videos, off the macroblock grid: 12 (chance ~0).
Phase-aligned across all pairs (min cos > 0.8): the 3 strongest pairs

| (ky, kx) | f (cycles/px) | period | coherence white / cilia / Veo | min pair cos |
|---|---|---|---|---|
| (126, 14) | (0.175, 0.011) | 5.7 px | 1.00 / 0.77 / 0.68 | +0.94 |
| (210, -14) | (0.292, -0.011) | 3.4 px | 0.99 / 0.67 / 0.63 | +0.98 |
| (238, 14) | (0.331, 0.011) | 3.0 px | 0.98 / 0.59 / 0.59 | +0.91 |

## Interpretation and limits

- A fixed-phase, content-independent luma pattern is present in all three Google videos and
  absent from two stock-x264 720p controls. That rules out generic H.264 at 720p as the source.
- It does **not** yet rule out a deterministic Google-side stage other than SynthID (Veo decoder /
  upscaler, Google's own encoder). Separating those needs a Veo output without SynthID, which is
  not normally obtainable; the practical next check is stability of these bins across many more
  Veo/Gemini videos and other resolutions (e.g. 1080p, portrait 720x1280).
- Key is 1280x720-specific (bin indices), as with the still-image codebooks.
