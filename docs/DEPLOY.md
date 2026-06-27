# Deployment notes — taking the API public

The dashboard API (`src/sentinel/api/app.py`) is read-only apart from one
inference route (`POST /map-techniques`, the zero-shot technique mapper). That
route runs SecureBERT, so it's the only meaningful resource-exhaustion vector.
Everything below exists so a public deployment **degrades gracefully (429/503)
rather than running the box out of memory**.

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

The rate limiter and concurrency cap are **single-process, in-memory**. With
multiple workers each gets its own counters — fine for a small deployment, but
for real limits put them at the reverse proxy (below).

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
