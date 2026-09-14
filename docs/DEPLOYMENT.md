# SignalScope — Deployment Guide

Two services:

| Service | Stack | Host | 
|---|---|---|
| Backend | FastAPI + PyTorch (EfficientNet-B0 ensemble) | **Render** (Docker) |
| Frontend | React + Vite + Tailwind (static SPA) | **Vercel** |

The frontend is a static SPA that calls the backend over HTTPS. In production
the browser talks to the Render service directly (`VITE_API_BASE_URL`), and
the backend already sends `Access-Control-Allow-Origin: *`, so no proxy layer
is needed.

---

## 1. Backend on Render

### Option A — Blueprint (fastest)

1. Push this repository to GitHub.
2. In [Render](https://render.com): **New → Blueprint**, connect the repo.
3. Render reads [`render.yaml`](../render.yaml) and creates the
   `signalscope-api` web service (Docker, health check `/health`).
4. When the deploy finishes, copy the service URL, e.g.
   `https://signalscope-api.onrender.com`.

### Option B — Manual

1. **New → Web Service**, connect the repo.
2. Runtime: **Docker** (it will use the repo-root `Dockerfile`).
3. Branch: the branch you deploy (e.g. `main`).
4. Health check path: `/health`.
5. Plan: **Starter** (or higher). See the resource note below.
6. **Create Web Service** and wait for the build (installs PyTorch — a few
   minutes).

> **Resource note.** The backend loads two EfficientNet-B0 models plus PyTorch,
> OpenCV and NumPy at startup (~600 MB+ RSS). Render's **free** tier (512 MB,
> 0.1 CPU) can OOM during a request; the **Starter** plan (512 MB, 0.5 CPU,
> $7/mo) is the safe minimum. Free-tier services also spin down after ~15
> minutes of inactivity — see §3 for the keep-alive cron.

**Verify** (from anywhere):

```bash
curl https://signalscope-api.onrender.com/health
# → {"status":"healthy","model_loaded":true,"device":"cpu","architecture":"EfficientNet-B0 ensemble (2)"}

curl -F "file=@some_image.jpg" https://signalscope-api.onrender.com/predict
```

---

## 2. Frontend on Vercel

1. Push the repository to GitHub.
2. In [Vercel](https://vercel.com): **Add New → Project**, import the repo.
3. **Framework preset:** Vite (auto-detected — see
   [`frontend/vercel.json`](../frontend/vercel.json)).
4. **Root Directory:** `frontend`.
5. **Environment Variables** (Project → Settings → Environment Variables):

   | Name | Value |
   |---|---|
   | `VITE_API_BASE_URL` | `https://signalscope-api.onrender.com` |

   (No trailing slash. This is baked in at build time — set it **before** the
   first build, or redeploy after changing it.)

6. **Deploy.** Vercel runs `npm run build` and serves `frontend/dist`.

**Verify:** open the Vercel URL. The header should show **Model online** (green
dot). Upload an image and confirm a verdict + Grad-CAM tabs render.

> **How the API URL works.** `frontend/src/api.ts` reads `VITE_API_BASE_URL`.
> When it is empty, the app uses relative URLs (the Vite dev proxy handles them
> locally). When set, every request — `/predict`, `/health`, and the
> `/files/...` visualization images — is prefixed with the backend URL. Blob
> previews (the local thumbnail) are left untouched.

---

## 3. Health endpoint + keep-alive cron

**Health endpoint:** `GET /health` on the backend returns:

```json
{"status":"healthy","model_loaded":true,"device":"cpu","architecture":"EfficientNet-B0 ensemble (2)"}
```

It is a cheap liveness check (the model is loaded once at startup, not per
request), so it is the right target for cron pings and Render's own health
check.

Render's free tier sleeps after ~15 minutes without traffic. Ping `/health`
every ≤10 minutes to keep the demo warm. Pick **one** of:

### A. GitHub Actions (template included)

The workflow template lives at [`deploy/keepalive.yml`](../deploy/keepalive.yml).
It pings the endpoint every 10 minutes. To enable it:

1. Copy it to `.github/workflows/keepalive.yml` (GitHub Actions only reads that
   directory) and push from an account with permission to write workflow files.
2. GitHub repo → **Settings → Secrets and variables → Actions → New repository
   secret**, name `SIGNALSCOPE_HEALTH_URL`, value
   `https://signalscope-api.onrender.com/health`.
3. The schedule activates once the file is on the default branch.

> Some CI/agent tokens are not allowed to create `.github/workflows/*` files
> (the `workflows` permission is required). That is why the file ships at
> `deploy/keepalive.yml` — copy it into place as above.

### B. Render Cron Jobs

Render → your service → **Cron Jobs** → add a job hitting
`https://signalscope-api.onrender.com/health` on the interval you want
(availability depends on your plan).

### C. External free monitors

- [cron-job.org](https://cron-job.org) — free, min. 1-minute interval, GET
  request to `/health`.
- [UptimeRobot](https://uptimerobot.com) — free, 5-minute interval monitor.

---

## 4. Local sanity check before deploying

```bash
# backend
.venv/bin/uvicorn app.api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health

# frontend (build as Vercel would)
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run build   # production build
npm run dev                                             # or dev server (proxies :8000)
```

---

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| Frontend header shows **Model offline** | Backend not up, or `VITE_API_BASE_URL` wrong/missing → check the env var and redeploy |
| `POST /predict` fails with CORS error | Backend CORS is `*`; make sure you hit the Render URL, not `localhost` |
| Render deploy succeeds but requests 502/503 | Cold start or OOM → upgrade plan, or hit `/health` to warm it first |
| Grad-CAM images broken but verdict shows | `VITE_API_BASE_URL` not set → visualization `/files/...` resolves against Vercel instead of Render |
| Upload > 15 MB rejected | Hard limit (config `inference.max_upload_mb`); expected behaviour |
