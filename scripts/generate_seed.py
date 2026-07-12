"""Build a read-only SQLite seed of the whole knowledge graph for the cloud demo.

Runs the full pipeline (OSINT ingest -> NLP enrich -> flow replay -> WAF replay)
into a single SQLite file. That file is published as a GitHub Release asset and
baked into the deploy image (Dockerfile.deploy), so the public dashboard serves a
real graph with no managed Postgres. See docs/DEPLOY.md ("Deploy to the cloud").

Needs the full env (ML stack + the CIC-IDS2017 dataset under data/cicids2017),
so it runs on a workstation, not in the slim deploy image. Usage:

    python scripts/generate_seed.py --out data/sentinel-seed.db
"""

import argparse
import os
import time
from pathlib import Path


def _banner(msg: str) -> None:
    print(f"\n===== {msg} ({time.strftime('%H:%M:%S')}) =====", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/sentinel-seed.db"))
    parser.add_argument("--nvd-days", type=int, default=2, help="NVD lookback window")
    parser.add_argument("--nvd-cap", type=int, default=800, help="max NVD CVEs to ingest")
    parser.add_argument("--replay-sample", type=int, default=300_000, help="flows for replay")
    parser.add_argument("--max-alerts", type=int, default=600, help="per detector")
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")  # libomp (lightgbm + torch)
    out = args.out.resolve()
    os.environ["SENTINEL_DATABASE_URL"] = f"sqlite:///{out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        out.unlink()

    import lightgbm  # noqa: F401  # load before torch on macOS

    from sentinel.config import get_settings

    get_settings.cache_clear()
    settings = get_settings()

    from sentinel.db import models  # noqa: F401  # register tables on Base.metadata
    from sentinel.db.base import Base, make_engine, session_scope

    _banner("create schema")
    Base.metadata.create_all(make_engine())

    from sentinel.ingest.attack import fetch_attack_techniques
    from sentinel.ingest.kev import fetch_kev_catalog
    from sentinel.ingest.nvd import fetch_recent_cves
    from sentinel.ingest.rss import fetch_rss_reports

    _banner("ingest ATT&CK techniques")
    with session_scope() as s:
        count = sum(bool(s.merge(t)) for t in fetch_attack_techniques())
    print("attack techniques:", count, flush=True)

    _banner("ingest CISA KEV")
    with session_scope() as s:
        count = sum(bool(s.merge(e)) for e in fetch_kev_catalog())
    print("kev entries:", count, flush=True)

    _banner("ingest RSS/Atom CTI feeds")
    with session_scope() as s:
        count = sum(bool(s.merge(r)) for r in fetch_rss_reports())
    print("rss reports:", count, flush=True)

    _banner(f"ingest NVD ({args.nvd_days}-day window, capped at {args.nvd_cap})")
    with session_scope() as s:
        count = 0
        for vuln in fetch_recent_cves(days=args.nvd_days):
            s.merge(vuln)
            count += 1
            if count >= args.nvd_cap:
                break
    print("nvd cves:", count, flush=True)

    _banner("enrich: technique tagging")
    from sentinel.nlp.encoders import BiEncoder
    from sentinel.nlp.mapper import TechniqueMapper, load_technique_docs
    from sentinel.nlp.tagging import tag_untagged_reports

    with session_scope() as s:
        mapper = TechniqueMapper(
            load_technique_docs(s),
            encoder=BiEncoder(),
            cache_dir=settings.nlp_embedding_cache_dir,
            model_name=settings.nlp_bi_encoder_model,
            lexical=True,
        )
        edges = tag_untagged_reports(
            s,
            mapper,
            method=settings.nlp_bi_encoder_model,
            top_k_per_sentence=settings.nlp_tag_top_k,
            min_score=settings.nlp_tag_min_score,
            max_techniques=settings.nlp_tag_max_techniques,
        )
    print("report-technique edges:", edges, flush=True)

    _banner("enrich: campaign correlation")
    from sentinel.correlate.campaigns import build_campaigns, link_report_cves

    with session_scope() as s:
        print("report-cve edges:", link_report_cves(s), flush=True)
    with session_scope() as s:
        print("campaigns:", build_campaigns(s), flush=True)

    from sentinel.ids import replay, waf_replay

    _banner("flow replay (5-detector ensemble -> alerts)")
    print(
        replay.main(
            [
                "--sample",
                str(args.replay_sample),
                "--max-alerts",
                str(args.max_alerts),
                "--seed",
                str(args.seed),
            ]
        ),
        flush=True,
    )

    _banner("WAF replay (SQLi payload detector -> T1190 alerts)")
    print(waf_replay.main([]), flush=True)

    size_mb = out.stat().st_size / 1e6
    print(f"\nSEED DONE -> {out} ({size_mb:.1f} MB)", flush=True)


if __name__ == "__main__":
    main()
