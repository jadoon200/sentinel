"""Feature-aligned loader for cross-dataset generalization testing.

The honest question the within-dataset numbers can't answer: does a model
trained on CIC-IDS2017 detect the *same attack* on a *different network*?
CSE-CIC-IDS2018 is a different testbed, year, and CICFlowMeter version, and
its CSV columns use abbreviated names ("Tot Fwd Pkts" vs 2017's "Total Fwd
Packet"). `canonical_columns` normalizes both header conventions to a common
key so the intersection of flow features can be compared directly.

The literature finding this reproduces: NIDS models scoring ~0.97 within a
dataset collapse toward chance across networks (arXiv:2402.10974). Brute-force
is the natural probe — 2017's FTP/SSH-Patator and 2018's FTP/SSH-BruteForce
are the same attack under different label strings.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

# 2018→2017 label aliasing: same attack, different campaign naming.
LABEL_ALIASES = {
    "FTP-BRUTEFORCE": "FTP-PATATOR",
    "SSH-BRUTEFORCE": "SSH-PATATOR",
    "DOS ATTACKS-HULK": "DOS HULK",
    "DOS ATTACKS-GOLDENEYE": "DOS GOLDENEYE",
    "DOS ATTACKS-SLOWLORIS": "DOS SLOWLORIS",
    "DDOS ATTACKS-LOIC-HTTP": "DDOS",
}

_DROP = re.compile(r"[\s_/]")


def canonical(name: str) -> str:
    """Map either dataset's column name to a convention-free key."""
    c = name.lower().strip()
    for a, b in (
        ("total", "tot"),
        ("packets", "pkt"),
        ("packet", "pkt"),
        ("pkts", "pkt"),
        ("length", "len"),
        ("bytes", "byt"),
        ("byts", "byt"),
        ("segment", "seg"),
        ("bulk", "blk"),
        ("count", "cnt"),
        ("backward", "bwd"),
        ("forward", "fwd"),
    ):
        c = c.replace(a, b)
    return _DROP.sub("", c)


def canonical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Rename a flow frame's columns to canonical keys (label kept as 'label')."""
    renamed = {col: canonical(col) for col in frame.columns}
    out = frame.rename(columns=renamed)
    if "label" not in out.columns:
        raise ValueError("no Label column found after canonicalization")
    return out


def load_2018_day(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False, skipinitialspace=True)
    frame.columns = frame.columns.str.strip()
    return canonical_columns(frame)


def normalize_label(label: str) -> str:
    up = " ".join(str(label).upper().split())
    return LABEL_ALIASES.get(up, up)


def shared_feature_xy(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, "pd.Series[int]", "pd.Series[int]", "pd.Series[str]"]:
    """Align two canonicalized frames to shared numeric features.

    Returns X_train, X_test, y_train (binary), y_test (binary), and the test
    labels (aliased to 2017 names). Identifier-like canonical columns are
    dropped, matching the within-dataset loader's leakage policy.
    """
    drop = {"flowid", "srcip", "dstip", "srcport", "dstport", "timestamp", "protocol", "label"}
    train_feats = set(train.columns) - drop
    test_feats = set(test.columns) - drop
    shared = sorted(train_feats & test_feats)

    def prep(frame: pd.DataFrame) -> tuple[pd.DataFrame, "pd.Series[int]", "pd.Series[str]"]:
        labels = frame["label"].map(normalize_label)
        x = frame[shared].apply(pd.to_numeric, errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan)
        y = (labels.str.upper() != "BENIGN").astype(int)
        return x, y, labels

    x_train, y_train, _ = prep(train)
    x_test, y_test, labels_test = prep(test)
    return x_train, x_test, y_train, y_test, labels_test
