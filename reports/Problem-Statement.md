# SignalScope — Problem Statement

*SIH 2026 — Problem Statement*

## 1. Problem Statement

Generative AI models can now produce photorealistic images in seconds. This is a huge creative
unlock, but it also creates a serious trust problem: synthetic images are increasingly used for
misinformation, fraud, fake product listings, and manipulated "evidence." As these tools become
more accessible, the line between a real photograph and a machine-generated one is disappearing —
and most people, platforms, and fact-checkers have no reliable way to tell the difference.

The core difficulty is not just building a real-vs-fake classifier. It is building one that
**generalizes**. A detector trained on outputs from one generator (say, Stable Diffusion) often
fails badly on images from a newer or different generator it has never seen — exactly the
situation that matters in the real world, since new generative models appear constantly. A
detector that only performs well on familiar generators gives a false sense of security.

There is a second, equally important problem: a verdict alone is not enough to build trust. If a
system simply says "AI-generated" with no reasoning, a journalist, platform moderator, or everyday
user has no way to judge whether to believe it, and no way to explain the decision to someone else.
An opaque black-box verdict does not build trust — it just shifts the burden of doubt.

**SignalScope addresses both problems together:**
1. Classify an image as real or AI-generated with strong accuracy that holds up on unseen
   generators, not just the ones seen during training.
2. Explain *why* — pointing to the specific visual cues (texture artifacts, lighting
   inconsistencies, anatomical errors, etc.) behind the verdict, so the output is useful and
   verifiable rather than an unexplained label.

Framed responsibly: SignalScope reports a *likelihood* ("likely AI-generated"), never an
accusation, and is scoped strictly to detecting synthetic *imagery in general* — it does not
attempt face-swap deepfake detection of real individuals or adjudicate political claims.