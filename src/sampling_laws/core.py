"""Log-domain elementary-symmetric-polynomial sampler and inclusion moments.

The product-subset law is NOT the sequential weighted-without-replacement law.
Algorithms are exact in real arithmetic; this implementation uses float64.
"""

from numbers import Integral

import numpy as np


def _integer(value, name, minimum=0):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _weights(weights, k):
    raw = np.asarray(weights)
    if np.iscomplexobj(raw):
        raise ValueError("weights must be real")
    values = np.asarray(raw, dtype=float).copy()
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("weights must be a nonempty finite nonnegative vector")
    k = _integer(k, "k")
    if k > np.count_nonzero(values):
        raise ValueError("k exceeds the number of positive-weight items")
    return values, k


def _table(log_weights, k):
    """Row i contains log elementary symmetric sums of the first i weights."""
    table = np.full((len(log_weights) + 1, k + 1), -np.inf)
    table[:, 0] = 0
    for i, weight in enumerate(log_weights):
        table[i + 1, 1:] = np.logaddexp(table[i, 1:], weight + table[i, :-1])
    return table


class ProductSubset:
    """P(S) proportional to product(weights[S]), subject to |S|=k.

    Weights are odds parameters, not inclusion probabilities. Zero weights
    exclude items; infinite weights are deliberately unsupported. All public
    probability arrays are newly returned, and input weights are copied.
    """

    def __init__(self, weights, k):
        self.weights, self.k = _weights(weights, k)
        self.n = len(self.weights)
        positive = self.weights > 0
        self._positive_prefix = np.r_[0, np.cumsum(positive)]
        logs = np.full(self.n, -np.inf)
        logs[positive] = np.log(self.weights[positive])
        self._shift = float(np.max(logs[positive])) if positive.any() else 0.0
        self._logs = logs - self._shift
        self._prefix = _table(self._logs, self.k)
        self._suffix = _table(self._logs[::-1], self.k)[::-1]
        self._log_z = float(self._prefix[-1, self.k])
        self.log_normalizer = self._log_z + self.k * self._shift
        for array in (self.weights, self._positive_prefix, self._logs, self._prefix, self._suffix):
            array.setflags(write=False)

    def log_probability(self, subset):
        indices = list(subset)
        if len(indices) != self.k or any(isinstance(i, bool) or not isinstance(i, Integral)
                                        or not 0 <= i < self.n for i in indices):
            raise ValueError("subset must contain k valid integer indices")
        if len(set(indices)) != len(indices):
            raise ValueError("subset contains duplicate indices")
        return float(self._logs[indices].sum() - self._log_z)

    def log_inclusion_probabilities(self):
        result = np.full(self.n, -np.inf)
        if self.k == 0:
            return result
        if self.k == np.count_nonzero(self.weights):
            result[self.weights > 0] = 0
            return result
        for i in range(self.n):
            if self.weights[i] > 0:
                # Split k-1 selected items between the prefix and suffix.
                excluded = np.logaddexp.reduce(self._prefix[i, :self.k]
                                               + self._suffix[i + 1, :self.k][::-1])
                result[i] = min(0.0, self._logs[i] + excluded - self._log_z)
        return result

    def inclusion_probabilities(self):
        return np.exp(self.log_inclusion_probabilities())

    def pair_inclusion_probabilities(self):
        """Joint inclusion matrix; diagonal is pi_i (not zero).

        Recompute conditional designs after fixing each selected item. This
        costs O(n^2*k), but avoids unstable subtraction of nearly equal weights.
        """
        marginal = self.inclusion_probabilities()
        joint = np.diag(marginal)
        if self.k < 2:
            return joint
        for i in range(self.n):
            if marginal[i] > 0:
                others = np.arange(self.n) != i
                conditional = ProductSubset(self.weights[others], self.k - 1)
                joint[i, others] = marginal[i] * conditional.inclusion_probabilities()
        return joint

    def sample(self, draws, rng):
        """Return (draws,n) Boolean inclusion masks, each containing exactly k."""
        draws = _integer(draws, "draws")
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        result = np.zeros((draws, self.n), dtype=bool)
        remaining = np.full(draws, self.k, dtype=int)
        for i in range(self.n - 1, -1, -1):
            active = np.flatnonzero(remaining > 0)
            if not len(active) or self.weights[i] == 0:
                continue
            slots = remaining[active]
            log_p = self._logs[i] + self._prefix[i, slots - 1] - self._prefix[i + 1, slots]
            probability = np.exp(np.minimum(log_p, 0))
            probability[slots == self._positive_prefix[i + 1]] = 1.0
            selected = active[rng.random(len(active)) < probability]
            result[selected, i] = True
            remaining[selected] -= 1
        if np.any(remaining):
            raise ArithmeticError("sampler failed its fixed-size invariant")
        return result

    def ht_totals(self, values, masks):
        """Horvitz-Thompson totals for this design; reject unsupported targets."""
        values = np.asarray(values)
        if np.iscomplexobj(values) or values.shape != (self.n,) or not np.all(np.isfinite(values)):
            raise ValueError("values must be a finite real vector matching the population")
        values = values.astype(float)
        masks = np.asarray(masks)
        if (masks.dtype.kind != "b" or masks.ndim != 2 or masks.shape[1] != self.n
                or np.any(masks.sum(axis=1) != self.k) or np.any(masks[:, self.weights == 0])):
            raise ValueError("masks must be Boolean supported fixed-size samples")
        marginal = self.inclusion_probabilities()
        if np.any((marginal == 0) & (values != 0)):
            raise ValueError("nonzero target has zero or numerically underflowed inclusion probability")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            corrected = np.divide(values, marginal, out=np.zeros_like(values), where=marginal > 0)
            totals = masks @ corrected
        if not np.all(np.isfinite(totals)):
            raise ArithmeticError("nonfinite inverse-probability estimate")
        return totals


def gumbel_top_k(weights, k, draws, rng):
    """Plackett-Luce sampling with order discarded, NOT product-subset sampling."""
    weights, k = _weights(weights, k)
    draws = _integer(draws, "draws")
    if not isinstance(rng, np.random.Generator):
        raise ValueError("rng must be a numpy.random.Generator")
    masks = np.zeros((draws, len(weights)), dtype=bool)
    if k == 0 or draws == 0:
        return masks
    logs = np.full(len(weights), -np.inf)
    logs[weights > 0] = np.log(weights[weights > 0])
    scores = logs + rng.gumbel(size=masks.shape)
    indices = np.argpartition(scores, -k, axis=1)[:, -k:]
    np.put_along_axis(masks, indices, True, axis=1)
    return masks
