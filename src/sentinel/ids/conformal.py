"""Conformal alert thresholds with label-free online budget control.

Adapted from the conformal toolkit in the author's time-series forecasting
project (split-conformal quantiles + Adaptive Conformal Inference). Two
deliberate changes make it fit NIDS instead of forecast intervals:

1. **One-sided p-values.** Forecasting calibrates two-sided intervals; an
   anomaly detector needs P(benign score >= observed). The split-conformal
   p-value  p = (1 + #{calibration >= s}) / (n_cal + 1)  is distribution-free
   and finite-sample: alerting when p <= alpha bounds FPR at alpha under
   exchangeability — replacing the ad-hoc p99 percentile with a guarantee.

2. **Label-free adaptation.** ACI corrects its running alpha from interval
   *misses*, which requires ground truth at every step. Live NIDS never has
   that, so the online controller regulates the **alert rate** instead:
   alpha_t+1 = alpha_t + gamma * (alpha - alert_t). When attacks are rare the
   alert rate approximates the FPR, recovering ACI's behaviour; when attacks
   surge, the budget spends itself on the most anomalous flows — bounded
   analyst load either way. (The measured failure this fixes: thresholds
   calibrated to 1% on Mon-Wed benign drift to 6-12% FPR on Thu-Fri.)

The O(log n) sorted-insert online quantile follows the original ACI
implementation; numpy searchsorted vectorizes the batch p-values.
"""

from bisect import bisect_left, insort
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


def conformal_pvalues(
    calibration: NDArray[np.float64], scores: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Split-conformal p-value of each score against benign calibration scores."""
    cal = np.sort(calibration)
    n_cal = len(cal)
    greater_equal = n_cal - np.searchsorted(cal, scores, side="left")
    return np.asarray((1.0 + greater_equal) / (n_cal + 1.0), dtype=np.float64)


@dataclass(frozen=True)
class BudgetResult:
    alerts: NDArray[np.bool_]
    alpha_path: NDArray[np.float64]
    threshold_path: NDArray[np.float64]


class AlertBudgetController:
    """Online alert-rate control over a time-ordered score stream.

    Seeds its score memory with the benign calibration set, thresholds each
    step at the (1 - alpha_t) empirical quantile of everything seen so far,
    then nudges alpha_t toward the target alert budget.
    """

    def __init__(
        self,
        calibration: NDArray[np.float64],
        alpha: float = 0.01,
        gamma: float = 0.005,
        adapt_memory: bool = True,
    ) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self.gamma = gamma
        self.adapt_memory = adapt_memory
        self._seen: list[float] = sorted(float(s) for s in calibration)
        self._alpha_t = alpha

    def _quantile(self) -> float:
        n = len(self._seen)
        clamped = min(max(self._alpha_t, 1e-6), 1.0)
        k = int(np.ceil((n + 1) * (1.0 - clamped)))
        return self._seen[min(max(k, 1), n) - 1]

    def step(self, score: float) -> tuple[bool, float]:
        """Process one score; returns (alert, threshold used)."""
        threshold = self._quantile()
        alert = score > threshold
        self._alpha_t += self.gamma * (self.alpha - float(alert))
        if self.adapt_memory and not alert:
            # Non-alerting scores are the stream's evolving "normal" — folding
            # them in is what tracks benign drift (attacks stay excluded,
            # mirroring the benign-only calibration assumption).
            if len(self._seen) > 200_000:
                del self._seen[0 : len(self._seen) // 2 : 2]
            insort(self._seen, float(score))
        return alert, threshold

    def run(self, scores: NDArray[np.float64]) -> BudgetResult:
        alerts = np.zeros(len(scores), dtype=bool)
        alphas = np.zeros(len(scores))
        thresholds = np.zeros(len(scores))
        for t, score in enumerate(scores):
            alert, threshold = self.step(float(score))
            alerts[t] = alert
            alphas[t] = self._alpha_t
            thresholds[t] = threshold
        return BudgetResult(alerts=alerts, alpha_path=alphas, threshold_path=thresholds)


def budget_alerts(
    calibration: NDArray[np.float64],
    scores: NDArray[np.float64],
    percentile: float,
    gamma: float = 0.005,
) -> NDArray[np.bool_]:
    """One-sided budget-controlled alert mask for a score stream — the drop-in
    replacement for ``scores > percentile(calibration, p)``.

    Targets the same (100 - ``percentile``)% nominal alert rate a fixed threshold
    aims for, but adapts online so the realized rate holds under benign drift
    instead of inflating (see ``AlertBudgetController``). For detectors whose
    score is one-sided (higher = more anomalous) and processed in stream order.
    """
    alpha = min(max((100.0 - percentile) / 100.0, 1e-6), 0.999)
    return AlertBudgetController(calibration, alpha=alpha, gamma=gamma).run(scores).alerts


def empirical_fpr(alerts: NDArray[np.bool_], is_benign: NDArray[np.bool_]) -> float:
    benign_alerts = alerts[is_benign]
    return float(benign_alerts.mean()) if len(benign_alerts) else 0.0


def rank_of(score: float, sorted_scores: list[float]) -> int:
    """Helper for diagnostics: how many calibration scores sit below `score`."""
    return bisect_left(sorted_scores, score)
