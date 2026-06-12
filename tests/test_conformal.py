import numpy as np

from sentinel.ids.conformal import AlertBudgetController, conformal_pvalues, empirical_fpr


def test_pvalues_bound_fpr_under_exchangeability() -> None:
    rng = np.random.default_rng(13)
    calibration = rng.normal(0, 1, 5000)
    benign_test = rng.normal(0, 1, 5000)  # same distribution: exchangeable

    p = conformal_pvalues(calibration, benign_test)

    # Finite-sample guarantee: P(p <= alpha) <= alpha (small slack for sampling noise).
    for alpha in (0.01, 0.05):
        assert (p <= alpha).mean() <= alpha * 1.5


def test_pvalues_flag_shifted_scores() -> None:
    rng = np.random.default_rng(13)
    calibration = rng.normal(0, 1, 5000)
    attacks = rng.normal(6, 1, 200)

    assert (conformal_pvalues(calibration, attacks) <= 0.01).mean() > 0.95


def test_budget_controller_holds_alert_rate_under_benign_drift() -> None:
    rng = np.random.default_rng(13)
    calibration = rng.normal(0, 1, 20_000)
    # Benign distribution shifts upward at test time — the failure mode that
    # pushed the static p99 threshold from 1% to 6-12% FPR on Thu-Fri.
    drifted_benign = rng.normal(1.0, 1.2, 30_000)

    static_threshold = np.percentile(calibration, 99)
    static_fpr = (drifted_benign > static_threshold).mean()

    result = AlertBudgetController(calibration, alpha=0.01, gamma=0.005).run(drifted_benign)
    controlled_rate = result.alerts.mean()

    assert static_fpr > 0.05  # static threshold blows the budget under drift
    assert controlled_rate < 0.02  # controller holds it near the 1% target


def test_budget_controller_still_catches_attacks_amid_drift() -> None:
    rng = np.random.default_rng(13)
    calibration = rng.normal(0, 1, 20_000)
    benign = rng.normal(0.8, 1.1, 5_000)
    attacks = rng.normal(8, 1, 50)
    stream = np.concatenate([benign[:4000], attacks, benign[4000:]])
    is_benign = np.ones(len(stream), dtype=bool)
    is_benign[4000 : 4000 + 50] = False

    result = AlertBudgetController(calibration, alpha=0.01, gamma=0.005).run(stream)

    assert result.alerts[~is_benign].mean() > 0.95  # attacks still alert
    assert empirical_fpr(result.alerts, is_benign) < 0.02
