# SignalScope — Demo Data

Safe, generic demo imagery for showcasing the detector. **No identifiable
individuals are included.**

## Structure

```
demo/
├── ai_generated/        # AI-generated demo images (produced for this project)
│   ├── ai_ceramic_mug.jpg
│   ├── ai_landscape.jpg
│   └── ai_abstract_art.jpg
├── real/                # REAL photographs (populate from public datasets)
├── degraded/            # degraded versions of the above (produced by the
│                        #   robustness module or manually)
└── README.md
```

## How the demo images were produced

- `ai_generated/*.jpg` were generated with a commercial image-generation
  model for demonstration purposes. They are labelled **AI-generated** and
  are generic scenes/objects (no people).
- `real/` is intentionally left empty in the repository. Populate it with
  genuinely real photographs, e.g. copy a few REAL images from the public
  CIFAKE dataset (`data/raw/cifake/test/REAL/`) or from the Defactify real
  set, or use your own photographs. **Do not place identifiable individuals
  here.**
- `degraded/` can be produced by running the robustness benchmark or with:
  `python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg --robustness`

## Run the demo

```bash
# Analyse an AI-generated demo image
python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg

# Analyse a real image (after you add one)
python -m app.inference demo/real/your_photo.jpg
```
