# SignalScope — Demo Video Script (3–5 minutes)

Record with the web app running (backend `uvicorn app.api:app --port 8000`,
frontend `cd frontend && npm run dev`). Keep it **under 5 minutes**.

## 0. Intro (0:00–0:20)

- "SignalScope — Telling Real From Synthetic."
- One line: binary real-vs-AI classification with a confidence score and a
  visual explanation, built for SIH 2026 Problem Statement 2.

## 1. Core task (0:20–1:30)

1. Open the **Analyze** page. Point out the **Model online** indicator.
2. Drop `demo/ai_generated/ai_ceramic_mug.jpg`.
   - Show the verdict **Likely AI-generated** + confidence bar + P(AI) / P(real).
   - Flip through **Original → Heatmap → Evidence overlay** tabs.
3. Drop a real photograph (any of your own, or a REAL image from the public
   CIFAKE test split `data/raw/cifake/test/REAL/`).
   - Show **Likely real** and that the model does NOT flag it.
4. Show the **Inconclusive** band quickly if you have a borderline image, or
   just mention that 0.20–0.80 maps to Inconclusive.

## 2. Explanation (1:30–2:20)

- On the AI mug result, read the "Why this verdict" evidence bullets aloud.
- Stress: "this is a likelihood, not proof" — point at the disclaimer line and
  the uncertainty sentence.

## 3. Robustness (2:20–3:00) — Bonus C

- In the CLI run:
  `python -m app.inference demo/ai_generated/ai_ceramic_mug.jpg --robustness`
- Show the table of P(AI) under JPEG / resize / blur / noise / screenshot and
  note where the verdict flips. Reference
  `reports/robustness_demo_results.csv`.

## 4. Provenance (3:00–3:30) — Bonus D

- Show the **Provenance signals** panel: EXIF present/absent and Content
  Credentials presence, with the line "Missing metadata is not evidence of AI
  generation."

## 5. Honest limitations (3:30–4:00)

- Open **Insights** and **About**: unseen-generator evaluation is pending;
  Grad-CAM is an explanation aid, not forensic proof; no people, no political
  claims.

## 6. Outro (4:00–4:30)

- Recap: confidence-based verdict + Grad-CAM + robustness + provenance, with
  responsible language throughout.

> Publish unlisted (e.g. YouTube/Drive) and paste the link in `README.md`
> under **Demo**.
