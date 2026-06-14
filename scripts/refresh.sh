#!/usr/bin/env bash
# Refresh the SENTINEL knowledge graph end-to-end: ingest OSINT, enrich with
# ATT&CK techniques + campaigns, then replay flows into intel-fused alerts.
#
# Designed to be cron/launchd-friendly: it activates the conda env, logs to
# data/refresh.log, and is safe to run repeatedly (every step is idempotent —
# ingest upserts, enrichment only tags untagged reports, replay rebuilds alerts).
#
# Enable a daily run (example, 06:00) without editing this file:
#   crontab -e
#   0 6 * * *  /Users/jayden/Documents/sentinel/scripts/refresh.sh >> /tmp/sentinel-cron.log 2>&1
#
# Requires: Postgres up (`make up`) and the `sentinel` conda env installed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# Activate the conda env (cron has a bare environment).
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate sentinel
fi

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "ingest: OSINT sources"
python -m sentinel.ingest.flows

log "enrich: ATT&CK technique tagging + campaign correlation"
python -m sentinel.ingest.flows enrich

log "replay: flows -> intel-fused alerts"
python -m sentinel.ids.replay

log "refresh complete"
