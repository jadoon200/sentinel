import numpy as np

from sentinel.nlp.lexical import BM25, reciprocal_rank_fusion, tokenize


def test_bm25_ranks_exact_term_matches_first() -> None:
    docs = [
        "PowerShell. Adversaries abuse powershell commands and scripts.",
        "Phishing. Emails with malicious links.",
        "Scheduled Task. Persistence via cron and schtasks.",
    ]
    bm25 = BM25(docs)

    scores = bm25.scores("the actor ran schtasks for persistence")

    assert scores.argmax() == 2
    assert scores[1] == 0.0  # no shared terms


def test_rrf_rewards_agreement_across_rankings() -> None:
    dense = np.array([0.9, 0.5, 0.1])
    lexical = np.array([2.0, 8.0, 1.0])

    fused = reciprocal_rank_fusion([dense, lexical])

    # doc0: ranks 1+2, doc1: ranks 2+1 — tie; doc2 last everywhere
    assert fused[2] < fused[0]
    assert abs(fused[0] - fused[1]) < 1e-9


def test_tokenize_lowercases_and_splits() -> None:
    assert tokenize("CVE-2026-1234 via PowerShell!") == ["cve", "2026", "1234", "via", "powershell"]
