"""Ensemble coverage: which attack families the detector ensemble catches, by whom.

Single-model recall is the wrong unit to judge the platform on — the autoencoder's
0.268 *overall* is modest because it is unsupervised on unseen families, but you
deploy the **ensemble**, not the autoencoder alone. Each detector specializes;
this runs all five on the temporal split (train Mon-Wed, test the unseen Thu-Fri
families) and reports, per family, the best detector and its recall.

It is a union *lower bound* (true union recall >= the best single detector), at
each detector's own benign-calibrated operating point — so the combined alert
rate is higher than any one detector's, the usual ensemble trade. The story:
every unseen Thu-Fri family is covered at ~0.7-1.0 by its specialist, even though
no single model covers them all. SQLi is the one exception flow data cannot see
(max robust-z 1.0 vs benign HTTP) — handled by the payload detector (`make sqli`),
a different modality, not this flow ensemble.

All five run on the **same temporal split** (train Mon-Wed, test the unseen
Thu-Fri families) at ~1% benign FPR, so the comparison is apples-to-apples — the
supervised model is calibrated on benign traffic (`--calibrate-fpr 0.01`) rather
than its default threshold, which collapses on unseen families by construction.

Each detector runs in its **own process**: LightGBM, MLX and torch vendor
conflicting native libraries that segfault when trained together in one process.

Run (heavy — trains the full ensemble): python scripts/eval_ensemble.py
"""

import argparse
import re
import subprocess
import sys

DETECTORS = ["train", "anomaly", "sequence", "profile", "beacon"]
LABELS = {
    "train": "supervised",
    "anomaly": "autoencoder",
    "sequence": "sequence",
    "profile": "profile",
    "beacon": "beacon",
}
# Supervised needs the temporal split + benign calibration to be on the same
# unseen-family, ~1%-FPR footing as the unsupervised detectors.
ARGS = {"train": ["--split", "temporal", "--calibrate-fpr", "0.01"]}
# --conformal applies only to the one-sided budget-controllable detectors;
# sequence (two-sided) and beacon (per-channel set) keep their percentile.
CONFORMAL_DETECTORS = {"anomaly", "profile"}
_RECALL = re.compile(r"^recall__(.+?):\s*([0-9.]+)\s*$", re.MULTILINE)


def _family(raw: str) -> str:
    name = raw.replace("_", " ")
    return "Bot" if name.startswith("Bot") else name


def main() -> dict[str, tuple[str, float]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--conformal",
        action="store_true",
        help="gate the one-sided detectors (autoencoder, profile) with the budget controller",
    )
    args = parser.parse_args()
    if args.conformal:
        print("ensemble mode: conformal budget control on autoencoder + profile\n")

    per_family: dict[str, list[tuple[str, float]]] = {}
    for module in DETECTORS:
        extra = ["--conformal"] if (args.conformal and module in CONFORMAL_DETECTORS) else []
        print(f"running {LABELS[module]} ({module})...", flush=True)
        result = subprocess.run(
            [sys.executable, "-m", f"sentinel.ids.{module}", *ARGS.get(module, []), *extra],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: {module} exited {result.returncode}\n{result.stderr[-500:]}")
            continue
        for raw, value in _RECALL.findall(result.stdout):
            per_family.setdefault(_family(raw), []).append((LABELS[module], float(value)))

    print("\n=== Ensemble coverage (best detector per unseen family) ===")
    best: dict[str, tuple[str, float]] = {}
    for family in sorted(per_family):
        detector, recall = max(per_family[family], key=lambda dr: dr[1])
        best[family] = (detector, recall)
        others = ", ".join(f"{d}={r:.2f}" for d, r in sorted(per_family[family]))
        print(f"  {family:28} best: {detector:11} {recall:.3f}   [{others}]")
    covered = sum(1 for _, recall in best.values() if recall >= 0.5)
    print(f"\n{covered}/{len(best)} families covered at recall >= 0.50 by their specialist")
    return best


if __name__ == "__main__":
    main()
