"""Load the corrected CIC-IDS2017 flow CSVs (Engelen et al., WTMC 2021).

The corrected dataset fixes >20% of the original flows (labeling, flow
termination, feature extraction) and adds " - Attempted" labels for attack
flows that never carried a payload. Identifier-like columns (IPs, ports,
timestamps, flow IDs) are dropped so the model can't shortcut-learn the
testbed topology — a known cause of inflated CIC-IDS2017 scores.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

LABEL_COLUMN = "Label"
DAY_COLUMN = "__day"  # capture day from the source filename for temporal splits

# Identity / topology columns: trivially separable in the testbed, useless in
# any other network. Both original and corrected header variants listed.
ID_COLUMNS = [
    "Flow ID",
    "Src IP",
    "Source IP",
    "Dst IP",
    "Destination IP",
    "Src Port",
    "Source Port",
    "Dst Port",
    "Destination Port",
    "Timestamp",
    "Attempted Category",
    DAY_COLUMN,
]

AttemptedPolicy = Literal["drop", "benign", "malicious"]


def load_flows(data_dir: Path, sample: int | None = None, seed: int = 13) -> pd.DataFrame:
    """Concatenate all day CSVs under data_dir, optionally downsampled."""
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSV files in {data_dir} — download the corrected dataset")
    frames = []
    for path in files:
        frame = pd.read_csv(path, low_memory=False, skipinitialspace=True)
        frame.columns = frame.columns.str.strip()
        frame[DAY_COLUMN] = path.stem.split("-")[0]
        frames.append(frame)
    flows = pd.concat(frames, ignore_index=True)
    if sample is not None and sample < len(flows):
        flows = flows.sample(n=sample, random_state=seed).reset_index(drop=True)
    return flows


def make_xy(
    flows: pd.DataFrame, attempted: AttemptedPolicy = "drop"
) -> tuple[pd.DataFrame, "pd.Series[int]", "pd.Series[str]"]:
    """Split flows into features X, binary target y, and original labels.

    `attempted` controls flows labeled "<attack> - Attempted" (attack ran but no
    payload was delivered): "drop" excludes them (default — they are neither
    clean benign traffic nor successful attacks), "benign"/"malicious" keeps
    them with the chosen target.
    """
    labels = flows[LABEL_COLUMN].astype(str).str.strip()
    is_attempted = labels.str.upper().str.endswith("- ATTEMPTED")

    if attempted == "drop":
        keep = ~is_attempted
        flows, labels = flows.loc[keep], labels.loc[keep]
        malicious = labels.str.upper() != "BENIGN"
    elif attempted == "benign":
        malicious = (labels.str.upper() != "BENIGN") & ~is_attempted
    else:
        malicious = labels.str.upper() != "BENIGN"

    features = flows.drop(columns=[LABEL_COLUMN, *[c for c in ID_COLUMNS if c in flows.columns]])
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.replace([np.inf, -np.inf], np.nan)
    return features, malicious.astype(int), labels
