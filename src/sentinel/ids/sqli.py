"""Application-layer SQL-injection detector — payload inspection, not netflow.

CIC-IDS2017 has only 12 SQLi flows, none in training. On basic volume/timing
features they look like benign HTTP, so the unsupervised flow detectors miss them
entirely (autoencoder/sequence/profile recall 0); a calibrated supervised model
does flag the 12 from the full feature set, but on 12 within-dataset flows and
only as "attack-ish," not "SQLi" (see docs/EVAL.md). What SQLi actually needs is
a detector that recognizes the attack by its signature — the SQL string in the
request payload, which CICFlowMeter never captures — and works on real requests.
So this is a different modality: a character n-gram classifier over request
payloads, the application-layer / WAF analogue of the flow IDS. Detections map to
T1190 (Exploit Public-Facing App), sharing the ATT&CK graph with every signal.

Validated the SENTINEL way — **cross-corpus**: train on one public payload corpus
and test on a different one, to prove the detector generalizes beyond a single
dataset's quirks rather than memorizing it (the same bar as the IDS cross-dataset
eval). Data (free, public; cached under data/sqli/, gitignored):

- HttpParamsDataset (Morzeux): real HTTP parameter values, attack_type-labeled.
- Kaggle SQLiV2: collected SQL statements / payloads, 0/1 labeled.

Usage:
    python -m sentinel.ids.sqli
"""

import argparse
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tenacity import retry, stop_after_attempt, wait_exponential

from sentinel.config import get_settings

TECHNIQUES = ["T1190"]  # Exploit Public-Facing Application
_USER_AGENT = "Mozilla/5.0 (compatible; SENTINEL-CTI/0.1)"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=2, max=20))
def _download(url: str, dest: Path) -> None:
    with httpx.Client(timeout=60, follow_redirects=True, headers={"User-Agent": _USER_AGENT}) as c:
        response = c.get(url)
        response.raise_for_status()
    # Decode with the response's detected charset (one corpus ships as UTF-16),
    # then cache as UTF-8 so the loader reads both corpora uniformly.
    dest.write_text(response.text, encoding="utf-8")


def _ensure_corpora(data_dir: Path) -> dict[str, Path]:
    """Download the corpora to data_dir on first use; return name -> path."""
    data_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, url in get_settings().sqli_corpus_urls.items():
        dest = data_dir / f"{name}.csv"
        if not dest.exists():
            _download(url, dest)
        paths[name] = dest
    return paths


def load_corpora(data_dir: Path | None = None) -> dict[str, tuple[list[str], NDArray[np.int_]]]:
    """Load each SQLi corpus as (payload texts, binary labels) — 1 = SQLi.

    Both sources are normalized to the same shape so either can train and the
    other can test it, which is the cross-corpus generalization protocol.
    """
    data_dir = data_dir or get_settings().sqli_data_dir
    paths = _ensure_corpora(data_dir)
    corpora: dict[str, tuple[list[str], NDArray[np.int_]]] = {}

    # HttpParamsDataset: keep only SQLi-vs-benign (drop xss/path/cmd) for a clean
    # SQLi task; column `attack_type` in {norm, sqli, ...}.
    http = pd.read_csv(paths["httpparams"], encoding_errors="ignore")
    http = http[http["attack_type"].isin(["sqli", "norm"])].dropna(subset=["payload"])
    corpora["httpparams"] = (
        http["payload"].astype(str).tolist(),
        np.asarray((http["attack_type"] == "sqli").to_numpy(), dtype=np.int_),
    )

    # Kaggle SQLiV2: columns Sentence, Label (1 = SQLi).
    kag = pd.read_csv(paths["sqliv2"], encoding_errors="ignore").dropna(subset=["Sentence"])
    corpora["sqliv2"] = (
        kag["Sentence"].astype(str).tolist(),
        np.asarray(
            pd.to_numeric(kag["Label"], errors="coerce").fillna(0).to_numpy(), dtype=np.int_
        ),
    )
    return corpora


def build_detector(c: float = 4.0) -> Any:
    """Character n-gram TF-IDF + logistic regression.

    Char n-grams (with word boundaries) capture SQL syntax — quotes, comment
    markers, `union select`, `or 1=1` — and transfer across payload styles far
    better than word tokens, which is what makes the cross-corpus recall hold.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(1, 3), min_df=3, sublinear_tf=True
                ),
            ),
            ("clf", LogisticRegression(max_iter=2000, C=c, class_weight="balanced")),
        ]
    )


def _metrics(
    y_true: NDArray[np.int_], proba: NDArray[np.float64], threshold: float = 0.5
) -> dict[str, float]:
    from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, (proba >= threshold).astype(int), average="binary", zero_division=0
    )
    return {
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _proba(detector: Any, texts: list[str]) -> NDArray[np.float64]:
    return np.asarray(detector.predict_proba(texts))[:, 1]


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args(argv)

    import mlflow
    from sklearn.model_selection import train_test_split

    settings = get_settings()
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("ids-sqli")

    corpora = load_corpora(args.data_dir)
    metrics: dict[str, float] = {}

    # Within-corpus, averaged over seeds (the easy number — same data distribution).
    for name, (texts, labels) in corpora.items():
        texts_arr = np.asarray(texts, dtype=object)
        accum: list[dict[str, float]] = []
        for seed in range(args.seeds):
            x_tr, x_te, y_tr, y_te = train_test_split(
                texts_arr, labels, test_size=0.3, random_state=seed, stratify=labels
            )
            det = build_detector().fit(list(x_tr), y_tr)
            accum.append(_metrics(y_te, _proba(det, list(x_te))))
        for key in accum[0]:
            metrics[f"within_{name}__{key}"] = float(np.mean([a[key] for a in accum]))

    # Cross-corpus (the honest test): train on one source, test on the other.
    names = list(corpora)
    for train_name in names:
        for test_name in names:
            if train_name == test_name:
                continue
            tr_texts, tr_y = corpora[train_name]
            te_texts, te_y = corpora[test_name]
            det = build_detector().fit(tr_texts, tr_y)
            for key, value in _metrics(te_y, _proba(det, te_texts)).items():
                metrics[f"cross_{train_name}_to_{test_name}__{key}"] = value

    with mlflow.start_run():
        mlflow.log_params({"analyzer": "char_wb(1,3)", "model": "logreg", "seeds": args.seeds})
        mlflow.log_metrics(metrics)

    for key, value in sorted(metrics.items()):
        print(f"{key}: {value:.4f}")
    return metrics


if __name__ == "__main__":
    main()
