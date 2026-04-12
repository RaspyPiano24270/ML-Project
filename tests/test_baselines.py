from src.evaluate_baselines import run_walkforward


def test_walkforward_baselines_smoke() -> None:
    results = run_walkforward(n_splits=3, max_splits=1)
    assert results["evaluated_folds"] == 1
    assert "classifier_accuracy" in results["avg_metrics"]
    assert "regressor_mae" in results["avg_metrics"]
