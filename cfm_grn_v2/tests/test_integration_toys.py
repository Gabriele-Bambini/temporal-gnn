import pytest

from cfm_grn_v2.experiments.run_synthetic import main


@pytest.mark.slow
def test_quick_toys_run_produces_verdicts(tmp_path):
    v = main(["--out", str(tmp_path), "--quick", "--seeds", "2", "--search-seeds", "1", "--datasets", "toys",
              "--max-configs", "2"])
    assert set(v) == {"toy_translation", "toy_variance", "toy_bimodal", "toy_condition_shift"}
    assert (tmp_path / "report.md").exists() and (tmp_path / "results.csv").exists()
