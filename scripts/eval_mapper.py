"""Evaluate the ATT&CK technique mapper against TRAM's human-annotated sentences.

TRAM bootstrap data: 11,130 sentences labeled with 50 techniques by human
analysts (center-for-threat-informed-defense/tram, Apache-2.0).

Usage (inside the sentinel conda env):
    python scripts/eval_mapper.py --sample 2000 [--rerank]

Reports hit@k: fraction of sentences where a gold technique appears in the
top-k mapper predictions (exact ID, and parent-technique level).
"""

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import httpx

from sentinel.config import get_settings
from sentinel.ingest.attack import fetch_attack_techniques
from sentinel.nlp.encoders import BiEncoder, CrossEncoderScorer
from sentinel.nlp.mapper import TechniqueMapper, technique_doc

TRAM_URL = (
    "https://raw.githubusercontent.com/center-for-threat-informed-defense"
    "/tram/main/data/training/bootstrap-training-data.json"
)
CACHE = Path("data/tram_bootstrap.json")


def load_tram_sentences() -> list[dict[str, Any]]:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TRAM_URL, timeout=120.0)
        response.raise_for_status()  # don't cache a 404/error body as if it were data
        CACHE.write_bytes(response.content)
    sentences = json.loads(CACHE.read_text())["sentences"]
    return [s for s in sentences if s.get("mappings") and len(s["text"].split()) >= 4]


def parent(technique_id: str) -> str:
    return technique_id.split(".")[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=2000)
    parser.add_argument("--rerank", action="store_true")
    parser.add_argument(
        "--procedures", action="store_true", help="enrich docs with procedure examples"
    )
    parser.add_argument("--hybrid", action="store_true", help="BM25 + dense rank fusion")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    sentences = load_tram_sentences()
    random.Random(args.seed).shuffle(sentences)
    sentences = sentences[: args.sample]
    print(
        f"evaluating on {len(sentences)} TRAM sentences "
        f"(rerank={args.rerank}, procedures={args.procedures}, hybrid={args.hybrid})"
    )

    techniques = fetch_attack_techniques()
    docs = [technique_doc(t, include_procedures=args.procedures) for t in techniques]
    settings = get_settings()
    reranker = CrossEncoderScorer() if args.rerank else None
    mapper = TechniqueMapper(
        docs,
        encoder=BiEncoder(),
        reranker=reranker,
        cache_dir=settings.nlp_embedding_cache_dir,
        model_name=settings.nlp_bi_encoder_model,
        lexical=args.hybrid,
    )

    hits = {k: 0 for k in (1, 3, 5, 10)}
    parent_hits = {k: 0 for k in (1, 3, 5, 10)}
    t0 = time.time()
    for i, sentence in enumerate(sentences):
        gold = {m["attack_id"] for m in sentence["mappings"] if m.get("attack_id")}
        gold_parents = {parent(g) for g in gold}
        matches = mapper.map_text(sentence["text"], top_k=args.top_k)
        predicted = [m.technique_id for m in matches]
        for k in hits:
            top = predicted[:k]
            hits[k] += bool(gold & set(top))
            parent_hits[k] += bool(gold_parents & {parent(p) for p in top})
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{len(sentences)} ({time.time() - t0:.0f}s)")

    n = len(sentences)
    print(f"\n{'k':>4} {'hit@k':>8} {'parent hit@k':>14}")
    for k in sorted(hits):
        print(f"{k:>4} {hits[k] / n:>8.3f} {parent_hits[k] / n:>14.3f}")


if __name__ == "__main__":
    main()
