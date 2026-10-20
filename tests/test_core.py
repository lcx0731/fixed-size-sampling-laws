from fractions import Fraction
from math import comb

import numpy as np
import pytest

from sampling_laws import ProductSubset, gumbel_top_k
from sampling_laws.reference import inclusion, product_law, sequential_law, total_moments


@pytest.mark.parametrize("k", range(7))
def test_all_subset_probabilities_against_rational_enumeration(k):
    weights = [1, 2, 3, 4, 5, 7]
    model = ProductSubset(weights, k)
    law = product_law(weights, k)
    assert sum(law.values()) == 1
    for subset, probability in law.items():
        assert np.exp(model.log_probability(subset)) == pytest.approx(float(probability), rel=2e-13)
    np.testing.assert_allclose(model.inclusion_probabilities(), list(map(float, inclusion(law, 6))), atol=2e-14)


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_uniform_hypergeometric_moments(k):
    model = ProductSubset(np.ones(6), k)
    assert model.log_normalizer == pytest.approx(np.log(comb(6, k)), abs=1e-14)
    np.testing.assert_allclose(model.inclusion_probabilities(), k / 6, atol=1e-14)
    joint = model.pair_inclusion_probabilities()
    np.testing.assert_allclose(joint[~np.eye(6, dtype=bool)], k * (k - 1) / 30, atol=1e-14)


def test_joint_probabilities_and_ht_variance_against_exact_states():
    weights, values, k = [1, 2, 3, 4, 5, 7], [4, 9, 2, 14, 5, 21], 3
    law = product_law(weights, k)
    model = ProductSubset(weights, k)
    pi, joint = model.inclusion_probabilities(), model.pair_inclusion_probabilities()
    oracle = np.array([[float(sum((p for s, p in law.items() if i in s and j in s), Fraction(0)))
                        for j in range(6)] for i in range(6)])
    np.testing.assert_allclose(joint, oracle, atol=3e-14)
    np.testing.assert_allclose(joint, joint.T, atol=3e-14)
    covariance = joint - np.outer(pi, pi)
    np.testing.assert_allclose(covariance.sum(axis=1), 0, atol=4e-14)
    assert np.linalg.eigvalsh(covariance).min() > -1e-13
    assert np.max(covariance[~np.eye(6, dtype=bool)]) <= 0
    exact_mean, exact_variance = total_moments(law, values, inclusion(law, 6))
    assert exact_mean == sum(values)
    weighted = np.array(values) / pi
    assert weighted @ covariance @ weighted == pytest.approx(float(exact_variance), rel=1e-12)
    masks = np.array([[i in s for i in range(6)] for s in law])
    totals = model.ht_totals(values, masks)
    assert np.dot(list(map(float, law.values())), totals) == pytest.approx(sum(values))


def test_two_sampling_laws_have_exact_distinct_probabilities():
    product = product_law([1, 2, 4], 2)
    sequential = sequential_law([1, 2, 4], 2)
    assert product[(0, 1)] == Fraction(1, 7)
    assert sequential[(0, 1)] == Fraction(11, 105)
    assert sequential[(0, 2)] == Fraction(2, 7)
    assert sequential[(1, 2)] == Fraction(64, 105)
    assert sum(sequential.values()) == 1


def test_log_partition_gradient_and_hessian_match_inclusion_moments():
    logs = np.log([1., 2., 3., 5.])
    model = ProductSubset(np.exp(logs), 2)
    marginal = model.inclusion_probabilities()
    covariance = model.pair_inclusion_probabilities() - np.outer(marginal, marginal)
    for i in range(4):
        step = np.eye(4)[i] * 1e-4
        plus, minus = ProductSubset(np.exp(logs + step), 2), ProductSubset(np.exp(logs - step), 2)
        assert (plus.log_normalizer - minus.log_normalizer) / 2e-4 == pytest.approx(marginal[i], abs=1e-9)
        np.testing.assert_allclose((plus.inclusion_probabilities() - minus.inclusion_probabilities()) / 2e-4,
                                  covariance[:, i], atol=1e-9)


def test_scale_and_permutation_invariance():
    weights = np.array([1., 2., 3., 4.])
    order = np.array([3, 1, 0, 2])
    base = ProductSubset(weights, 2)
    scaled = ProductSubset(weights * 1e150, 2)
    reordered = ProductSubset(weights[order], 2)
    np.testing.assert_allclose(scaled.inclusion_probabilities(), base.inclusion_probabilities(), atol=1e-13)
    np.testing.assert_allclose(reordered.inclusion_probabilities(), base.inclusion_probabilities()[order], atol=1e-14)
    assert scaled.log_normalizer - base.log_normalizer == pytest.approx(2 * np.log(1e150))


@pytest.mark.parametrize("k", [0, 1, 2, 3])
def test_sample_size_zero_weights_and_reproducibility(k):
    weights = [0, 1, 0, 4, 9]
    model = ProductSubset(weights, k)
    masks = model.sample(100, np.random.default_rng(61))
    np.testing.assert_array_equal(masks, model.sample(100, np.random.default_rng(61)))
    np.testing.assert_array_equal(masks.sum(axis=1), k)
    assert not masks[:, [0, 2]].any()
    sequential = gumbel_top_k(weights, k, 100, np.random.default_rng(62))
    np.testing.assert_array_equal(sequential.sum(axis=1), k)
    assert not sequential[:, [0, 2]].any()
    assert model.sample(0, np.random.default_rng(0)).shape == (0, 5)


def test_sampler_empirical_subset_frequencies_match_their_own_laws():
    weights, k, draws = [1, 2, 4], 2, 30000
    cases = [(ProductSubset(weights, k).sample(draws, np.random.default_rng(1)), product_law(weights, k)),
             (gumbel_top_k(weights, k, draws, np.random.default_rng(2)), sequential_law(weights, k))]
    for masks, law in cases:
        for subset, probability in law.items():
            actual = np.mean(masks[:, subset].all(axis=1))
            p = float(probability)
            assert abs(actual - p) < 6 * np.sqrt(p * (1 - p) / draws) + 1 / draws


def test_underflow_preserves_log_probabilities_and_rejects_unsafe_ht():
    model = ProductSubset([1e-200, 1e200], 1)
    assert model.log_inclusion_probabilities()[0] == pytest.approx(-400 * np.log(10))
    assert model.inclusion_probabilities()[0] == 0
    with pytest.raises(ValueError, match="underflowed"):
        model.ht_totals([1, 1], np.array([[False, True]]))


def test_zero_weight_target_and_empty_design():
    model = ProductSubset([0, 2], 1)
    masks = np.array([[False, True]])
    assert model.ht_totals([0, 7], masks)[0] == 7
    with pytest.raises(ValueError):
        model.ht_totals([3, 7], masks)
    empty = ProductSubset([0, 0], 0)
    assert empty.log_probability([]) == 0
    np.testing.assert_array_equal(empty.inclusion_probabilities(), [0, 0])
    assert not empty.sample(5, np.random.default_rng(1)).any()


def test_input_copy_and_readonly_weight_storage():
    weights = np.array([1., 2., 3.])
    model = ProductSubset(weights, 1)
    weights[:] = 0
    np.testing.assert_allclose(model.inclusion_probabilities(), [1/6, 2/6, 3/6])
    with pytest.raises(ValueError):
        model.weights[0] = 10


@pytest.mark.parametrize("weights,k", [([], 0), ([[1, 2]], 1), ([1, -1], 1), ([1, np.nan], 1),
                                        ([1, np.inf], 1), ([1j], 1), ([0, 0], 1), ([1], 2),
                                        ([1], -1), ([1], True), ([1], 0.5)])
def test_invalid_design(weights, k):
    with pytest.raises(ValueError):
        ProductSubset(weights, k)


@pytest.mark.parametrize("subset", [[0], [0, 0], [0, 3], [-1, 1], [True, 1], [0., 1.]])
def test_invalid_subset(subset):
    with pytest.raises(ValueError):
        ProductSubset([1, 2, 3], 2).log_probability(subset)


@pytest.mark.parametrize("draws", [-1, 0.5, True])
def test_invalid_draws(draws):
    with pytest.raises(ValueError):
        ProductSubset([1, 2], 1).sample(draws, np.random.default_rng(1))


def test_invalid_rng_and_ht_masks():
    model = ProductSubset([1, 0, 2], 1)
    with pytest.raises(ValueError):
        model.sample(1, None)
    for masks in (np.array([[1, 0, 0]]), np.array([[True, True, False]]), np.array([[False, True, False]])):
        with pytest.raises(ValueError):
            model.ht_totals([1, 0, 2], masks)


def test_oracle_enumeration_limits_and_validation():
    with pytest.raises(ValueError):
        product_law([1] * 30, 15)
    with pytest.raises(ValueError):
        sequential_law([1] * 10, 8)
    with pytest.raises(ValueError):
        product_law([1., 2], 1)
    assert sequential_law([0, 2], 1) == {(0,): Fraction(0), (1,): Fraction(1)}


def test_categorical_and_complete_samples_agree_across_laws():
    assert product_law([1, 2, 4], 1) == sequential_law([1, 2, 4], 1)
    assert product_law([1, 2, 4], 3) == sequential_law([1, 2, 4], 3)
    assert product_law([1, 2, 4], 0) == sequential_law([1, 2, 4], 0)


def test_non_enumerable_population_keeps_probability_and_size_invariants():
    weights = np.exp(np.linspace(-10, 10, 300))
    model = ProductSubset(weights, 40)
    marginal = model.inclusion_probabilities()
    assert marginal.sum() == pytest.approx(40, abs=1e-10)
    assert np.all(np.diff(marginal) >= -1e-13)
    log_weights = np.log(weights)
    assert log_weights[:40].sum() <= model.log_normalizer
    assert model.log_normalizer <= log_weights[-40:].sum() + np.log(float(comb(300, 40)))
    masks = model.sample(128, np.random.default_rng(90))
    np.testing.assert_array_equal(masks.sum(axis=1), 40)
