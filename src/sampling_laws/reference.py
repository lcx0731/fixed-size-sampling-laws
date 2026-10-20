"""Small-population oracles using Fraction arithmetic, no dynamic programming."""

from fractions import Fraction
from itertools import combinations, permutations
from math import comb, perm, prod
from numbers import Integral


def _check(weights, k, ordered=False):
    weights = tuple(weights)
    if not weights or any(isinstance(w, bool) or not isinstance(w, Integral) or w < 0 for w in weights):
        raise ValueError("reference weights must be nonnegative integers")
    if isinstance(k, bool) or not isinstance(k, Integral) or not 0 <= k <= sum(w > 0 for w in weights):
        raise ValueError("invalid reference sample size")
    states = perm(len(weights), k) if ordered else comb(len(weights), k)
    if states > 100000:
        raise ValueError("enumeration limited to 100000 states")
    return tuple(map(int, weights))


def product_law(weights, k):
    weights = _check(weights, k)
    masses = {subset: prod(weights[i] for i in subset) for subset in combinations(range(len(weights)), k)}
    normalizer = sum(masses.values())
    return {subset: Fraction(mass, normalizer) for subset, mass in masses.items()}


def sequential_law(weights, k):
    weights = _check(weights, k, ordered=True)
    law = {subset: Fraction(0) for subset in combinations(range(len(weights)), k)}
    for order in permutations(range(len(weights)), k):
        probability, remaining = Fraction(1), sum(weights)
        for i in order:
            if weights[i] == 0:
                probability = Fraction(0)
                break
            probability *= Fraction(weights[i], remaining)
            remaining -= weights[i]
        law[tuple(sorted(order))] += probability
    return law


def inclusion(law, n):
    return [sum((p for subset, p in law.items() if i in subset), Fraction(0)) for i in range(n)]


def total_moments(law, values, denominators):
    """Exact moments for a weighted total using supplied positive denominators."""
    if len(values) != len(denominators) or any(d <= 0 for d in denominators):
        raise ValueError("one positive denominator per population value is required")
    estimates = {s: sum((Fraction(values[i]) / denominators[i] for i in s), Fraction(0)) for s in law}
    mean = sum((law[s] * value for s, value in estimates.items()), Fraction(0))
    variance = sum((law[s] * (value - mean)**2 for s, value in estimates.items()), Fraction(0))
    return mean, variance
