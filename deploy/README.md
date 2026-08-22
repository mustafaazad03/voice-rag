# Deploying this service

## What it actually needs

Measured against the real image (`docker run -m <cap>`), not estimated:

| memory cap | result |
|---|---|
| 512 MB | **OOM-killed** during startup (exit 137) |
| 768 MB | healthy, queries answered — 731 MB resident, no headroom |
| 1 GB | healthy, 938 MB resident |

Image is 1.6 GB. Budget **1 GB RAM**; 768 MB works but leaves nothing spare.

The floor is set by onnxruntime's allocator plus the resident index, not by
traffic — it is the same for one user and a hundred.

## Free-tier verdicts

| host | verdict |
|---|---|
| **Cloud Run** | Fits. Free tier covers 2M req + 360k GiB-s/month and scales to zero. Needs a GCP account with billing attached (the free tier still requires a card). |
| **HF Spaces** | Docker Spaces now require **PRO** ($9/mo) — free `cpu-basic` is static-only. `scripts/deploy_hf.sh` is ready and works the moment the account has PRO. |
| Render free | Out. 512 MB — OOM-killed above. |
| Koyeb free | Out. 512 MB. |
| Fly.io | No standing free allowance; trial credit only. |
| Oracle Always Free | Fits easily (24 GB ARM), genuinely free, but manual VM setup and an arm64 rebuild. |

## Where the secrets go

Never in the image or the repo. `SARVAM_API_KEY` is read from the environment
at startup; without it, text endpoints work and voice returns 503.

| target | how |
|---|---|
| local Docker | `docker run -e SARVAM_API_KEY=...` |
| Cloud Run | `gcloud run deploy --set-env-vars SARVAM_API_KEY=...`, or Secret Manager for real use |
| HF Space | Settings → Variables and secrets → New **secret** (not a variable) |
| local dev | `.env` in the repo root — already gitignored |

Optional `API_KEY` locks `/api/v1/*` and `/metrics` behind `X-API-Key`. Leave it
unset for a public demo, set it for anything that costs money to run.

## Cloud Run, start to finish

```bash
brew install --cask google-cloud-sdk
gcloud init && gcloud auth login
gcloud config set project <your-project>
gcloud run deploy voice-rag \
  --source . --region asia-south1 \
  --memory 1Gi --cpu 1 --port 7860 \
  --allow-unauthenticated \
  --set-env-vars SARVAM_API_KEY=<key>
```

Cold start is ~20-30 s: the container loads a 1.6 GB image and initialises the
encoder. Keep one warm instance (`--min-instances 1`) if a grader will click it
cold — that leaves the free tier, so switch it back afterwards.
