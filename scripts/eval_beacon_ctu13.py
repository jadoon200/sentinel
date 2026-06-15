"""Bulletproof the beacon-by-dispersion detector on CTU-13 — many botnet channels.

The dispersion detector closed the CIC-IDS2017 Bot gap (Bot 5/5 @1.6% FPR), but
on only *five* C2 channels of *one* botnet (ARES). The honest caveat was: validate
on a dataset with many beacon channels and real IPs. CTU-13 (Stratosphere) is
exactly that — 13 captures, seven botnet families (Neris, Rbot, Virut, Menti,
Sogou, Murlo, NSIS.ay), each a real infection with Src/Dst addresses retained.

Same detector, same rule (per (src→dst) channel, max robust-z over the coefficient
of variation of TotBytes and mean packet size; benign-calibrated, p99 threshold —
`ids/beacon.BeaconScorer`). Per scenario: calibrate on that capture's benign
channels, measure recall on its botnet channels. If dispersion separates C2 from
benign across families and networks, the 5-channel foothold becomes hundreds.

The dispersion signal is byte-size only — no timing, no IP/port as a feature — so
the no-topology-leak rule still holds.

Run (needs CTU-13 extracted under data/ctu13/CTU-13-Dataset/):
    python scripts/eval_beacon_ctu13.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from sentinel.config import get_settings
from sentinel.ids.beacon import STAT_NAMES, BeaconScorer

# CTU-13 scenario -> botnet family (Stratosphere documentation).
FAMILY = {
    1: "Neris",
    2: "Neris",
    9: "Neris",
    3: "Rbot",
    4: "Rbot",
    10: "Rbot",
    11: "Rbot",
    5: "Virut",
    13: "Virut",
    6: "Menti",
    7: "Sogou",
    8: "Murlo",
    12: "NSIS.ay",
}
MIN_EVENTS = 16


def _cv(values: np.ndarray) -> float:
    mean = float(np.nanmean(values))
    return float(np.nanstd(values) / mean) if mean > 0 else 0.0


def _channel_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, group in df.groupby(["src", "dst"], sort=False):
        if len(group) < MIN_EVENTS:
            continue
        # SrcBytes is the forward (host→C2) analog of CIC's forward-payload bytes,
        # the dimension that carried the ARES poll-vs-tasking signature; TotBytes
        # is kept as a second stat so the scorer keys on whichever disperses.
        rows.append(
            {
                "fwd_bytes_cv": _cv(group["src_bytes"].to_numpy(dtype=float)),
                "pkt_len_cv": _cv(group["bytes"].to_numpy(dtype=float)),
                "bot": bool(group["bot"].any()),
            }
        )
    return pd.DataFrame(rows)


def _load_scenario(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(
        path,
        usecols=["SrcAddr", "DstAddr", "TotPkts", "TotBytes", "SrcBytes", "Label"],
        low_memory=False,
    )
    return pd.DataFrame(
        {
            "src": raw["SrcAddr"].astype(str),
            "dst": raw["DstAddr"].astype(str),
            "pkts": pd.to_numeric(raw["TotPkts"], errors="coerce"),
            "bytes": pd.to_numeric(raw["TotBytes"], errors="coerce"),
            "src_bytes": pd.to_numeric(raw["SrcBytes"], errors="coerce"),
            "bot": raw["Label"].astype(str).str.contains("Botnet", case=False, na=False),
        }
    ).dropna(subset=["pkts", "bytes", "src_bytes"])


def main() -> None:
    root = get_settings().ids_data_dir.parent / "ctu13" / "CTU-13-Dataset"
    per_family: dict[str, list[float]] = {}
    total_bot = total_caught = 0
    print(f"{'scen':>4} {'family':<9} {'bot_ch':>7} {'recall':>8} {'fpr':>7} {'auc':>7}")
    for scenario in sorted(FAMILY):
        files = list((root / str(scenario)).glob("*.binetflow"))
        if not files:
            continue
        channels = _channel_stats(_load_scenario(files[0]))
        bot = channels[channels.bot]
        benign = channels[~channels.bot]
        if len(bot) == 0 or len(benign) < 50:
            continue
        scorer = BeaconScorer().fit(benign[STAT_NAMES].to_numpy(dtype=float))
        benign_scores = scorer.score(benign[STAT_NAMES].to_numpy(dtype=float))
        threshold = float(np.percentile(benign_scores, 99.0))
        all_scores = scorer.score(channels[STAT_NAMES].to_numpy(dtype=float))
        y = channels.bot.to_numpy().astype(int)
        bot_scores = scorer.score(bot[STAT_NAMES].to_numpy(dtype=float))
        recall = float((bot_scores > threshold).mean())
        fpr = float((benign_scores > threshold).mean())
        auc = roc_auc_score(y, all_scores) if y.any() and (y == 0).any() else float("nan")
        fam = FAMILY[scenario]
        per_family.setdefault(fam, []).append(recall)
        total_bot += len(bot)
        total_caught += int((bot_scores > threshold).sum())
        print(f"{scenario:>4} {fam:<9} {len(bot):>7} {recall:>8.3f} {fpr:>7.3f} {auc:>7.3f}")

    print("\nby family (mean recall @~1% FPR):")
    for fam in sorted(per_family):
        print(f"  {fam:<9} {np.mean(per_family[fam]):.3f}  ({len(per_family[fam])} scenarios)")
    if total_bot:
        print(
            f"\noverall: {total_caught}/{total_bot} botnet channels caught "
            f"({total_caught / total_bot:.3f}) across {len(per_family)} families"
        )


if __name__ == "__main__":
    main()
