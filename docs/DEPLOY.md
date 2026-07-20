# Deployment notes — taking the API public

The knowledge-graph surface in `src/sentinel/api/app.py` is read-only. The API
also has one stateless inference route (`POST /map-techniques`) and an optional,
isolated few-shot calibration lab. Calibration is **off by default**: unless
`SENTINEL_API_ENABLE_CALIBRATION=true`, every `/calibration/*` route returns
404 and no calibration table can be mutated. When enabled, its writes stay in
the three calibration-only tables; there is still no write path into the threat
knowledge graph.

The mapper and calibration retraining both run models. Everything below exists
so a public deployment **degrades gracefully (429/503) rather than running the
box out of memory**.

Two layers do the work: app-level guards (already in the code, tunable by env)
and infra-level controls (your job at deploy time). The app guards are
belt-and-suspenders — the reverse proxy is the primary defence.

## App-level (in code, configure via `SENTINEL_*` env vars)

| Env var | Default | Purpose |
| --- | --- | --- |
| `SENTINEL_API_ALLOWED_ORIGINS` | `""` | **Must set in prod.** Comma-separated exact origins for the deployed dashboard, e.g. `https://sentinel.example.com`. Empty keeps the localhost-only CORS regex used in dev — with it empty, a browser on your real domain is blocked from calling the API. |
| `SENTINEL_API_MAX_REQUEST_CHARS` | `20000` | Max characters of pasted text; longer → `422`. Also drives an early `413` body-size cut-off (≈4 bytes/char) before the body is buffered. |
| `SENTINEL_API_RATE_LIMIT_REQUESTS` | `30` | Per-client requests allowed per window on `/map-techniques`; over → `429`. |
| `SENTINEL_API_RATE_LIMIT_WINDOW_SECONDS` | `60` | The rate-limit window. |
| `SENTINEL_API_TRUST_FORWARDED_HEADER` | `false` | Derive the per-client rate-limit key from the first `X-Forwarded-For` hop. Leave **off** unless the API is behind a trusted proxy that sets it — on a directly-exposed server the header is client-controlled, so an attacker could rotate it to dodge the limit. Off → the socket peer IP is used. Set `true` when deploying behind the reverse proxy below. |
| `SENTINEL_API_INFERENCE_CONCURRENCY` | `2` | Hard cap on simultaneous model inferences; bounds peak RAM/CPU. Excess requests wait, then `503`. |
| `SENTINEL_API_INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `15` | How long a request waits for a free inference slot before `503`. |
| `SENTINEL_API_WARM_MODEL` | `false` | Set `true` in prod to warm the mapper in a background thread at startup so the first public request doesn't pay the ~20s model load. |
| `SENTINEL_API_ENABLE_CALIBRATION` | `false` | Enables the isolated `/calibration/*` workflow. Leave false unless a calibration pack is present and the host has been sized for LightGBM retraining. Disabled routes deliberately return `404`. |
| `SENTINEL_CALIBRATION_PACK_PATH` | `data/calibration-pack` | Directory pack produced by `make build-calibration-pack` (four Parquet files plus JSON metadata). |
| `SENTINEL_API_CALIBRATION_RATE_LIMIT_REQUESTS` | `120` | Per-client calibration requests per shared rate-limit window. The higher quota accommodates a 50-flow labelling loop; over → `429`. |

The rate limiters and concurrency cap are **single-process, in-memory**. With
multiple workers each gets its own counters — fine for a small deployment, but
for real limits put them at the reverse proxy (below).

## Optional local calibration lab

Build the frozen pack before enabling the routes:

```bash
make build-calibration-pack
```

The default `data/calibration-pack/` is a directory of Parquet frames plus
metadata for a representative CSE-CIC-IDS2018 **DoS** scenario: 100,000 source
rows, a 24,000-flow selectable pool, 8,000 target-benign calibration flows, and
a disjoint 48,000-flow held-out test. The source, pool, threshold-calibration,
and test roles are frozen separately. To build a broader pack instead:

```bash
python scripts/build_calibration_pack.py --families brute-force DoS Bot
```

Then set the opt-in flag (in `.env` or the process environment) and run the API
and dashboard normally:

```bash
SENTINEL_API_ENABLE_CALIBRATION=true make api
make ui
```

The isolated routes are:

| Route | Purpose |
| --- | --- |
| `POST /calibration/batches` | Seed and persist a reproducible batch (score-stratified by default). |
| `GET /calibration/batches/{id}` | Return batch progress, sampled flows, and completed runs. |
| `POST /calibration/flows/{id}/label` | Record the operator's answer, then reveal that flow's hidden ground truth. |
| `POST /calibration/flows/{id}/simulate-label` | Clearly marked demo shortcut that applies hidden ground truth. |
| `POST /calibration/batches/{id}/retrain` | Fit source plus operator-labelled target flows and grade once on the held-out test. |
| `GET /calibration/curve` | Return the frozen multi-family WS3 reference curve shown beside the live result. |

Rows are stored only in `calibration_batches`, `calibration_flows`, and
`calibration_runs`. Sampling is reproducible for a fixed strategy and seed,
re-labelling overwrites the prior answer, ground truth is withheld until the
operator answers, and retraining shares the inference concurrency semaphore so
a busy host returns `503` instead of piling up fits.

## Infra-level (your job at deploy)

- **Reverse proxy (nginx / Caddy)** — the primary defence:
  - `client_max_body_size` (nginx default is already 1 MB → rejects large
    uploads before they reach the app).
  - `limit_req` for robust, cross-worker rate limiting.
  - TLS termination (HTTPS). Set `SENTINEL_API_ALLOWED_ORIGINS` to the `https://`
    origin.
  - Forward the real client IP (`X-Forwarded-For`) and set
    `SENTINEL_API_TRUST_FORWARDED_HEADER=true` so the app keys per-client limits
    on the first hop. It is **ignored by default** — on a directly-exposed
    server the header is spoofable, so only trust it once a proxy sets it.
- **Don't expose the dev server.** Run uvicorn behind the proxy with a sane
  worker count (`uvicorn ... --workers N`), not bound to a public interface.
- **Sizing.** The mapper holds SecureBERT (~GB) in memory. Give the host enough
  RAM for `INFERENCE_CONCURRENCY` simultaneous inferences plus the model, or it
  will OOM regardless of the guards. Enable `SENTINEL_API_WARM_MODEL=true`.
- **Database** — Postgres stays on a private network; never expose it. Secrets
  via env only.

## Minimal prod env example

```bash
SENTINEL_API_ALLOWED_ORIGINS=https://sentinel.example.com
SENTINEL_API_WARM_MODEL=true
SENTINEL_API_TRUST_FORWARDED_HEADER=true  # behind a trusted proxy that sets X-Forwarded-For
# defaults are reasonable; tighten the rate limit if the host is small:
# SENTINEL_API_RATE_LIMIT_REQUESTS=10
# SENTINEL_API_INFERENCE_CONCURRENCY=1
```

## Deploy to the cloud (free, one service)

The section above is the full public-hardening story (rate limits, TLS, sizing).
For a **zero-cost portfolio demo** there's a much smaller path: one container that
serves the dashboard and the default-off/read-only API, with the graph baked in
as a SQLite file — no managed Postgres, no second service, no CORS.

**How it fits together**

- `Dockerfile.deploy` — multi-stage: builds the React dashboard, installs the
  slim API (`requirements-api.txt`, no ML stack), copies the built dashboard in,
  and pulls the seed DB. The API serves the SPA from its own origin
  (`SENTINEL_API_DASHBOARD_DIST`) and reads the graph from the bundled SQLite
  (`SENTINEL_DATABASE_URL=sqlite:///…`). It binds to `$PORT`.
- `render.yaml` — a Render Blueprint declaring that one free web service.
- The seed is a **read-only snapshot**, published as a GitHub Release asset (kept
  out of git) and fetched at image-build time.

Because the deploy image is the *slim* API, the live **“Try the mapper” panel
returns 503 by design** (SecureBERT isn't installed) — everything else (feed,
landscape, briefing, Navigator export) is fully served.

Calibration on this slim public image remains a **follow-up**, not a claimed
deployed feature. Before enabling it, publish the directory pack as a versioned
Release asset, teach the image build to fetch it, include and validate the
Parquet/LightGBM dependency set, and measure retraining RAM and latency on the
actual free Render tier. Keep `SENTINEL_API_ENABLE_CALIBRATION=false` until all
four checks pass.

**1. Generate + publish the seed** (needs the full env + `data/cicids2017/`):

```bash
python scripts/generate_seed.py --out data/sentinel-seed.db
gh release create seed-v1 data/sentinel-seed.db \
  --title "Dashboard seed v1" --notes "Read-only graph snapshot for the cloud demo"
```

Regenerating later (more feeds accumulated, or an OTX key set for richer
campaigns): rerun the script, publish under a **new tag** (`seed-v2`, …), and bump
the `ADD …/seed-v?/…` URL in `Dockerfile.deploy` so the image cache invalidates.

**2. Deploy on Render**

1. Push this repo to GitHub (public, so the image build can fetch the release asset).
2. Render → **New → Blueprint** → pick the repo. It reads `render.yaml` and
   creates the `sentinel` web service on the **free** plan.
3. First build takes a few minutes (npm build + pip install + seed fetch). When
   it's live, the dashboard is at `https://sentinel-XXXX.onrender.com`.

**3. Notes**

- **Cold starts.** Free Render web services sleep after ~15 min idle; the first
  hit then waits ~30–60 s. To keep it warm, point a free uptime monitor
  (e.g. UptimeRobot) at `/health` every 10 min — one always-pinged free service
  fits inside the monthly free hours.
- **Region.** `render.yaml` defaults to `singapore` (closest to the DIS audience);
  change it if you prefer.
- **Rate limiting behind Render's proxy.** The container only ever sees Render's
  load balancer as the TCP peer, so without `SENTINEL_API_TRUST_FORWARDED_HEADER`
  every visitor would share one rate-limit bucket (one busy client could 429
  everyone). `render.yaml` therefore sets it to `true` — safe here because
  Render's edge controls `X-Forwarded-For`; only flip it off if you move the
  image somewhere that exposes it directly to clients.
- **Refreshing data.** The seed is a point-in-time snapshot. Regenerate and bump
  the tag whenever you want the demo to reflect newer intel.
