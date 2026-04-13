import numpy as np
import pandas as pd

import src.evaluate_baselines as evaluate_baselines


def _mock_load_kaggle_csv(*, target_col=None, **_kwargs):
    rng = np.random.default_rng(7)
    n_rows = 60
    base = pd.DataFrame(
        {
            "Temperature": rng.normal(20.0, 5.0, n_rows),
            "Humidity": rng.uniform(35.0, 95.0, n_rows),
            "Wind Speed": rng.uniform(1.0, 20.0, n_rows),
        }
    )

    if target_col == evaluate_baselines.CLASSIFIER_TARGET:
        threshold = np.median(base["Humidity"])
        base[target_col] = (base["Humidity"] > threshold).astype(int)
    elif target_col == evaluate_baselines.REGRESSOR_TARGET:
        base[target_col] = (0.03 * base["Humidity"] + 0.01 * base["Wind Speed"]).astype(float)
    else:
        return None

    return base


def test_walkforward_baselines_smoke(monkeypatch) -> None:
    monkeypatch.setattr(evaluate_baselines, "load_kaggle_csv", _mock_load_kaggle_csv)
    results = evaluate_baselines.run_walkforward(n_splits=3, max_splits=1)
    assert results["evaluated_folds"] == 1
    assert "classifier_accuracy" in results["avg_metrics"]
    assert "regressor_mae" in results["avg_metrics"]
