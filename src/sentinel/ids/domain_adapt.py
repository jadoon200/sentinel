"""Domain adaptation for cross-network IDS transfer (label-free where possible).

The cross-dataset finding (docs/EVAL.md): a 2017-trained model ranks 2018
attacks (AUC 0.94) but its operating point collapses — the score distribution
shifted because the feature distributions differ between networks. These are
the standard, zero-cost attempts to close that gap, each measured rather than
assumed:

- `coral` — CORrelation ALignment (Sun & Saenko, 2016): a closed-form linear
  transform that matches the source features' covariance to the target's,
  using only unlabelled feature distributions.
- `feature_shift` / `stable_features` — rank features by how much their benign
  distribution moved between networks, keep the transfer-stable ones.
- `BenignQuantileTransform` — represent each flow feature by its percentile
  relative to the local benign population, so source and target networks share
  a distribution-free feature scale.
- `quantile_map` — transport source features into the target network's units by
  composing the source benign ECDF with the target benign inverse ECDF.
- `select_labels` — choose a target-network labelling batch without labels via
  blind random, uncertainty, score-stratified, coreset, or cluster sampling.

These methods need only target *benign* traffic — which any defender has on
their own network — so they stay within the project's label-free, zero-cost
rules.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.cluster import MiniBatchKMeans

from sentinel.ids.data import FlowScaler


def _impute(values: NDArray[np.float64], medians: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(np.isnan(values), medians, values)


class BenignQuantileTransform:
    """Map features to their midpoint-rank ECDF under a benign reference sample.

    NaNs are replaced with the corresponding benign median before ranking. A
    constant (including all-NaN) benign feature contains no rank information, so
    it maps to 0.5 for every input. Values outside the fitted benign range clamp
    naturally to 0 or 1.

    The transform is monotone within each feature: it preserves each feature's
    ordering while replacing incompatible cross-network units with a local
    benign-relative scale in ``[0, 1]``.
    """

    def __init__(self) -> None:
        self._n_features: int | None = None
        self._medians: NDArray[np.float64] | None = None
        self._sorted: list[NDArray[np.float64]] | None = None
        self._constant: NDArray[np.bool_] | None = None

    def fit(self, benign: pd.DataFrame) -> "BenignQuantileTransform":
        """Fit per-feature benign medians and sorted ECDF reference values."""
        values = benign.to_numpy(dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("benign reference must contain at least one row")

        n_features = values.shape[1]
        medians = np.empty(n_features, dtype=np.float64)
        sorted_columns: list[NDArray[np.float64]] = []
        constant = np.empty(n_features, dtype=np.bool_)

        for feature in range(n_features):
            column = values[:, feature]
            observed = column[~np.isnan(column)]
            # An all-NaN benign feature has no information. Giving it a finite
            # sentinel lets it follow the same constant-feature path without
            # emitting np.nanmedian's all-NaN warning.
            median = float(np.median(observed)) if len(observed) else 0.0
            fitted = np.sort(np.where(np.isnan(column), median, column))
            medians[feature] = median
            sorted_columns.append(fitted)
            constant[feature] = bool(fitted[0] == fitted[-1])

        self._n_features = n_features
        self._medians = medians
        self._sorted = sorted_columns
        self._constant = constant
        return self

    def transform(self, x: pd.DataFrame) -> NDArray[np.float64]:
        """Return midpoint-rank benign ECDF values for ``x``."""
        n_features, medians, sorted_columns, constant = self._fitted_state()
        values = x.to_numpy(dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != n_features:
            raise ValueError(f"expected {n_features} features, got {values.shape[1]}")

        result = np.empty(values.shape, dtype=np.float64)
        for feature, reference in enumerate(sorted_columns):
            if constant[feature]:
                result[:, feature] = 0.5
                continue
            column = np.where(np.isnan(values[:, feature]), medians[feature], values[:, feature])
            left = np.searchsorted(reference, column, side="left")
            right = np.searchsorted(reference, column, side="right")
            result[:, feature] = (left + right) / (2.0 * len(reference))
        return result

    def _inverse(self, quantiles: NDArray[np.float64]) -> NDArray[np.float64]:
        """Map quantiles back through the fitted benign empirical distribution."""
        n_features, _, sorted_columns, _ = self._fitted_state()
        if quantiles.ndim != 2 or quantiles.shape[1] != n_features:
            raise ValueError(f"expected {n_features} quantile features, got {quantiles.shape[1]}")

        result = np.empty(quantiles.shape, dtype=np.float64)
        for feature, reference in enumerate(sorted_columns):
            midpoint_ranks = (np.arange(len(reference), dtype=np.float64) + 0.5) / len(reference)
            result[:, feature] = np.interp(
                quantiles[:, feature],
                midpoint_ranks,
                reference,
                left=reference[0],
                right=reference[-1],
            )
        return result

    def _fitted_state(
        self,
    ) -> tuple[
        int,
        NDArray[np.float64],
        list[NDArray[np.float64]],
        NDArray[np.bool_],
    ]:
        if (
            self._n_features is None
            or self._medians is None
            or self._sorted is None
            or self._constant is None
        ):
            raise ValueError("transform must be fitted before use")
        return self._n_features, self._medians, self._sorted, self._constant


def quantile_map(
    source: pd.DataFrame,
    source_benign: pd.DataFrame,
    target_benign: pd.DataFrame,
) -> NDArray[np.float64]:
    """Transport source features into target units through their benign ECDFs.

    For each feature, this computes ``Q_target(ECDF_source(value))`` using
    linear interpolation over the sorted target benign values. The map is
    monotone per feature, so within-feature ranking is preserved while only the
    cross-feature scale changes. NaN medians are fitted independently on the
    two benign reference samples.
    """
    if source.shape[1] != source_benign.shape[1] or source.shape[1] != target_benign.shape[1]:
        raise ValueError("source and benign references must have the same feature count")
    source_transform = BenignQuantileTransform().fit(source_benign)
    target_transform = BenignQuantileTransform().fit(target_benign)
    return target_transform._inverse(source_transform.transform(source))


def _matrix_power_psd(matrix: NDArray[np.float64], power: float) -> NDArray[np.float64]:
    """Raise a symmetric PSD matrix to a real power via eigendecomposition."""
    vals, vecs = np.linalg.eigh(matrix)
    vals = np.clip(vals, 1e-8, None)
    return np.asarray((vecs * (vals**power)) @ vecs.T)


def coral(source: pd.DataFrame, target: pd.DataFrame, eps: float = 1e-3) -> NDArray[np.float64]:
    """Align source feature covariance to target; returns the transformed source.

    Label-free: uses only the two feature matrices. The target medians impute
    NaNs consistently so the covariance is well-defined.
    """
    medians = np.nanmedian(target.to_numpy(dtype=np.float64), axis=0)
    xs = _impute(source.to_numpy(dtype=np.float64), medians)
    xt = _impute(target.to_numpy(dtype=np.float64), medians)
    d = xs.shape[1]

    ms, mt = xs.mean(axis=0), xt.mean(axis=0)
    cs = np.cov(xs - ms, rowvar=False) + eps * np.eye(d)
    ct = np.cov(xt - mt, rowvar=False) + eps * np.eye(d)
    # whiten source, then recolour with the target covariance
    aligned = (xs - ms) @ _matrix_power_psd(cs, -0.5) @ _matrix_power_psd(ct, 0.5) + mt
    return np.asarray(aligned)


def feature_shift(source_benign: pd.DataFrame, target_benign: pd.DataFrame) -> "pd.Series[float]":
    """Per-feature standardized mean shift between the two benign populations.

    |mean_target - mean_source| / pooled_std — high means the feature's benign
    behaviour moved between networks and is unlikely to transfer.
    """
    s = source_benign.apply(pd.to_numeric, errors="coerce")
    t = target_benign.apply(pd.to_numeric, errors="coerce")
    pooled = np.sqrt((s.var() + t.var()) / 2.0).replace(0.0, np.nan)
    shift = (t.mean() - s.mean()).abs() / pooled
    result: pd.Series[float] = shift.fillna(0.0).sort_values(ascending=False)
    return result


def stable_features(
    source_benign: pd.DataFrame, target_benign: pd.DataFrame, keep_frac: float = 0.6
) -> list[str]:
    """The transfer-stable feature subset — those whose benign mean moved least."""
    shift = feature_shift(source_benign, target_benign)
    n_keep = max(1, int(len(shift) * keep_frac))
    return sorted(shift.index[-n_keep:].tolist())


def select_labels(
    pool_x: NDArray[np.float64],
    n: int,
    *,
    strategy: str,
    scores: NDArray[np.float64] | None = None,
    seed: int = 13,
) -> NDArray[np.intp]:
    """Select unique pool-row indices for labelling without using labels.

    ``random`` is retained as an alias of the deployable ``random-blind``
    selector. The evaluation harness special-cases its historical balanced
    ``random`` control because that oracle needs ground-truth labels and cannot
    honestly live behind this label-free interface.

    ``coreset`` performs k-center greedy selection in robust-scaled feature
    space; ``cluster`` chooses the nearest unique point to each mini-batch
    k-means centroid; ``stratified`` samples evenly across source-model score
    deciles; and ``active`` chooses scores closest to 0.5.
    """
    values = np.asarray(pool_x, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("pool_x must be a two-dimensional feature matrix")

    n_pool = len(values)
    if n <= 0 or n_pool == 0:
        return np.empty(0, dtype=np.intp)
    if n >= n_pool:
        return np.arange(n_pool, dtype=np.intp)

    valid = {"random", "random-blind", "active", "coreset", "cluster", "stratified"}
    if strategy not in valid:
        choices = ", ".join(sorted(valid))
        raise ValueError(f"unknown label-selection strategy {strategy!r}; choose from {choices}")

    rng = np.random.default_rng(seed)
    if strategy in {"random", "random-blind"}:
        return np.asarray(rng.choice(n_pool, n, replace=False), dtype=np.intp)

    if strategy in {"active", "stratified"}:
        if scores is None:
            raise ValueError(f"{strategy} selection requires source-model scores")
        score_values = np.asarray(scores, dtype=np.float64)
        if score_values.ndim != 1 or len(score_values) != n_pool:
            raise ValueError("scores must be one-dimensional with one value per pool row")
        if strategy == "active":
            uncertainty = np.abs(score_values - 0.5)
            return np.asarray(np.argsort(uncertainty, kind="stable")[:n], dtype=np.intp)
        return _stratified_indices(score_values, n, rng)

    frame = pd.DataFrame(values)
    scaled = np.asarray(FlowScaler().fit(frame).transform(frame), dtype=np.float64)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=10.0, neginf=-10.0)
    if strategy == "coreset":
        return _coreset_indices(scaled, n, rng)
    return _cluster_indices(scaled, n, seed)


def _stratified_indices(
    scores: NDArray[np.float64], n: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Draw evenly from equal-count score deciles (or fewer bins when n < 10)."""
    n_bins = min(10, n, len(scores))
    order = np.argsort(scores, kind="stable")
    bins = np.array_split(order, n_bins)
    per_bin, remainder = divmod(n, n_bins)
    selected: list[NDArray[np.intp]] = []
    for bin_number, bucket in enumerate(bins):
        take = per_bin + int(bin_number < remainder)
        selected.append(np.asarray(rng.choice(bucket, take, replace=False), dtype=np.intp))
    result = np.concatenate(selected).astype(np.intp, copy=False)
    rng.shuffle(result)
    return result


def _coreset_indices(
    scaled: NDArray[np.float64], n: int, rng: np.random.Generator
) -> NDArray[np.intp]:
    """Run k-center greedy with a running minimum-distance vector."""
    n_pool = len(scaled)
    candidate_limit = max(20_000, n)
    if n_pool > candidate_limit:
        candidates = np.asarray(rng.choice(n_pool, candidate_limit, replace=False), dtype=np.intp)
    else:
        candidates = np.arange(n_pool, dtype=np.intp)

    candidate_x = scaled[candidates]
    centroid = candidate_x.mean(axis=0)
    first = int(np.argmin(np.sum((candidate_x - centroid) ** 2, axis=1)))
    selected = np.empty(n, dtype=np.intp)
    selected[0] = first
    min_distance = np.sum((candidate_x - candidate_x[first]) ** 2, axis=1)
    min_distance[first] = -np.inf

    for position in range(1, n):
        farthest = int(np.argmax(min_distance))
        selected[position] = farthest
        distance = np.sum((candidate_x - candidate_x[farthest]) ** 2, axis=1)
        min_distance = np.minimum(min_distance, distance)
        min_distance[farthest] = -np.inf
    return candidates[selected]


def _cluster_indices(scaled: NDArray[np.float64], n: int, seed: int) -> NDArray[np.intp]:
    """Pick one unique pool point nearest each MiniBatchKMeans centroid."""
    model = MiniBatchKMeans(
        n_clusters=n,
        random_state=seed,
        batch_size=min(4096, len(scaled)),
        n_init=3,
    ).fit(scaled)
    centroids = np.asarray(model.cluster_centers_, dtype=np.float64)
    available = np.ones(len(scaled), dtype=np.bool_)
    selected = np.empty(n, dtype=np.intp)
    for position, centroid in enumerate(centroids):
        distances = np.sum((scaled - centroid) ** 2, axis=1)
        distances[~available] = np.inf
        nearest = int(np.argmin(distances))
        selected[position] = nearest
        available[nearest] = False
    return selected


def few_shot_training_set(
    source_x: pd.DataFrame,
    source_y: "pd.Series[int]",
    target_x: pd.DataFrame,
    target_y: "pd.Series[int]",
    n_labels: int = 50,
    seed: int = 13,
) -> tuple[pd.DataFrame, "pd.Series[int]"]:
    """Build the few-shot adapted training set: source + N labelled target flows.

    This is the measured cross-network fix (docs/EVAL.md): the label-free
    transforms above all failed, but folding a balanced handful of labelled
    target-network flows into the source training set re-anchors the decision
    boundary to the target's feature scale and recovers 0.95-0.99 recall across
    attack families. Draws n_labels // 2 attack + n_labels // 2 benign target
    flows; the caller trains any classifier on the returned (X, y).

    The target columns are aligned to the source's by intersection, so the two
    frames must share a (canonical) feature schema — see `cross_dataset`.
    """
    rng = np.random.default_rng(seed)
    shared = [c for c in source_x.columns if c in target_x.columns]
    atk_idx = np.flatnonzero(target_y.to_numpy() == 1)
    ben_idx = np.flatnonzero(target_y.to_numpy() == 0)
    half = n_labels // 2
    take = np.concatenate(
        [
            rng.choice(atk_idx, min(half, len(atk_idx)), replace=False),
            rng.choice(ben_idx, min(half, len(ben_idx)), replace=False),
        ]
    )
    x = pd.concat([source_x[shared], target_x.iloc[take][shared]], ignore_index=True)
    y = pd.concat(
        [source_y.reset_index(drop=True), target_y.iloc[take].reset_index(drop=True)],
        ignore_index=True,
    )
    return x, y
