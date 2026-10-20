import pytest

from sampling_laws.experiment import run


@pytest.mark.parametrize("draws", [0, 999, True, 1000.5])
def test_invalid_draws(tmp_path, draws):
    with pytest.raises(ValueError):
        run(tmp_path, draws)


def test_full_report_truth_law_mismatch_and_reproducibility(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    summary = run(first, draws=10000)
    run(second, draws=10000)
    assert (first / "summary.json").read_bytes() == (second / "summary.json").read_bytes()
    assert summary["population_total"] == 55
    assert summary["estimators"][0]["exact_mean_fraction"] == "55"
    assert summary["estimators"][3]["exact_mean_fraction"] == "55"
    assert abs(summary["estimators"][1]["exact_bias"]) > 0.1
    assert abs(summary["estimators"][2]["exact_bias"]) > 0.1
    assert summary["law_total_variation"] > 0.01
    for name in ("sampling-laws.png", "design-aware-estimation.png"):
        assert (first / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
