"""Compare sampling laws and exact design-based total-estimation moments."""

import argparse
from fractions import Fraction
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .core import ProductSubset, gumbel_top_k
from .reference import inclusion, product_law, sequential_law, total_moments


WEIGHTS = [1, 2, 3, 4, 5, 7]
VALUES = [4, 9, 2, 14, 5, 21]


def plots(output, summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    subset_records = summary["subsets"]
    x = np.arange(len(subset_records))
    for offset, design, color in [(-0.19, "product", "#257f8a"), (0.19, "sequential", "#c26b48")]:
        axes[0].bar(x + offset, [r[design] for r in subset_records], width=.36, color=color, alpha=.7, label=f"{design}: exact")
        axes[0].scatter(x + offset, [r[f"{design}_empirical"] for r in subset_records], color=color,
                        marker="_", s=60, zorder=4, label=f"{design}: sampled")
    axes[0].set(title="Same weights and sample size, different subset laws", ylabel="Probability",
                xticks=x, xticklabels=["".join(map(str, r["subset"])) for r in subset_records], xlabel="Selected indices (unordered)")
    axes[0].tick_params(axis="x", labelrotation=90, labelsize=8)
    axes[0].legend(fontsize=8)
    x = np.arange(len(WEIGHTS))
    for offset, key, label, color in [(-.25, "product", "Product inclusion", "#257f8a"),
                                     (0., "sequential", "Sequential inclusion", "#c26b48"),
                                     (.25, "naive", "k w / sum(w)", "#9383a8")]:
        axes[1].bar(x + offset, summary["inclusion"][key], width=.24, color=color, label=label)
    axes[1].set(title="Weights are not inclusion probabilities", ylabel="Probability / proposed denominator",
                xticks=x, xticklabels=[f"{i}\nw={w}" for i, w in enumerate(WEIGHTS)], ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    fig.savefig(output / "sampling-laws.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    records = summary["estimators"]
    x = np.arange(len(records))
    labels = ["Product\ncorrect pi", "Product\nnaive k*w/sum", "Sequential\nproduct pi", "Sequential\ncorrect pi"]
    axes[0].errorbar(x, [r["empirical_mean"] for r in records],
                     yerr=[2 * r["mean_mcse"] for r in records], fmt="o", color="#257f8a", capsize=5, label="Monte Carlo mean ± 2 MCSE")
    axes[0].scatter(x, [r["exact_mean"] for r in records], marker="_", s=350, color="#c26b48", label="Exact expectation")
    axes[0].axhline(sum(VALUES), ls=":", color="#444444", label="Population total")
    axes[0].set(title="Estimator bias depends on the sampling design", ylabel="Estimated population total", xticks=x, xticklabels=labels)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, [r["exact_rmse"] for r in records], color="#257f8a", alpha=.7, label="Exact RMSE")
    axes[1].scatter(x, [r["empirical_rmse"] for r in records], color="#c26b48", marker="_", s=300, label="Sample RMSE")
    axes[1].set(title="Bias and variance both matter", ylabel="RMSE about the true total", xticks=x, xticklabels=labels)
    axes[1].legend(fontsize=8)
    fig.savefig(output / "design-aware-estimation.png", dpi=160)
    plt.close(fig)


def run(output, draws=50000):
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1000:
        raise ValueError("draws must be an integer >= 1000")
    model, k = ProductSubset(WEIGHTS, 3), 3
    laws = {"product": product_law(WEIGHTS, k), "sequential": sequential_law(WEIGHTS, k)}
    pi = {name: inclusion(law, len(WEIGHTS)) for name, law in laws.items()}
    naive = [Fraction(k * w, sum(WEIGHTS)) for w in WEIGHTS]
    samples = {"product": model.sample(draws, np.random.default_rng(20261020)),
               "sequential": gumbel_top_k(WEIGHTS, k, draws, np.random.default_rng(20261021))}
    summary = {"config": {"weights": WEIGHTS, "values": VALUES, "k": k, "draws": draws,
                           "product_seed": 20261020, "sequential_seed": 20261021},
               "population_total": sum(VALUES), "subsets": [], "estimators": [],
               "inclusion": {**{name: list(map(float, p)) for name, p in pi.items()}, "naive": list(map(float, naive))}}
    np.testing.assert_allclose(model.inclusion_probabilities(), summary["inclusion"]["product"], atol=2e-14)
    for subset in laws["product"]:
        record = {"subset": list(subset)}
        for name, law in laws.items():
            p = float(law[subset])
            count = int(np.sum(samples[name][:, subset].all(axis=1)))
            frequency = count / draws
            if abs(frequency - p) > 6 * np.sqrt(p * (1 - p) / draws) + 1 / draws:
                raise ArithmeticError("sample frequency disagrees with its rational law")
            record.update({name: p, f"{name}_fraction": str(law[subset]),
                           f"{name}_count": count, f"{name}_empirical": frequency})
        summary["subsets"].append(record)
    specs = [("product", "correct", pi["product"]), ("product", "naive", naive),
             ("sequential", "wrong_product", pi["product"]), ("sequential", "correct", pi["sequential"])]
    for design, correction, denominator in specs:
        mean, variance = total_moments(laws[design], VALUES, denominator)
        mean_float, var_float = float(mean), float(variance)
        estimates = samples[design] @ (np.array(VALUES) / np.array(list(map(float, denominator))))
        mcse = np.sqrt(var_float / draws)
        if abs(estimates.mean() - mean_float) > 6 * mcse:
            raise ArithmeticError("estimated mean disagrees with its exact design expectation")
        if design == "product" and correction == "correct":
            np.testing.assert_allclose(estimates, model.ht_totals(VALUES, samples[design]), atol=1e-11)
        bias = mean_float - sum(VALUES)
        record = {"design": design, "correction": correction, "exact_mean": mean_float,
                  "exact_mean_fraction": str(mean), "exact_variance": var_float,
                  "exact_variance_fraction": str(variance), "exact_bias": bias,
                  "exact_rmse": float(np.sqrt(var_float + bias**2)), "mean_mcse": float(mcse),
                  "empirical_mean": float(estimates.mean()), "empirical_sd": float(estimates.std(ddof=1)),
                  "empirical_rmse": float(np.sqrt(np.mean((estimates - sum(VALUES))**2)))}
        summary["estimators"].append(record)
        print(f"{design}/{correction}: exact mean={mean_float:.6f}, sampled mean={estimates.mean():.6f}", flush=True)
    summary["law_total_variation"] = float(sum(abs(laws["product"][s] - laws["sequential"][s])
                                                   for s in laws["product"]) / 2)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n")
    plots(output, summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--draws", type=int, default=50000)
    args = parser.parse_args()
    run(args.output, args.draws)


if __name__ == "__main__":
    main()
