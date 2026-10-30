import pandas as pd
import pytest

from assumptions.ExploratoryDataChecker import ExploratoryDataChecker


def test_run_all_checks_with_numeric_column(rng, capsys):
    values = list(rng.normal(size=50)) + [50.0]
    checker = ExploratoryDataChecker(pd.DataFrame({"x": values}))
    checker.run_all_checks()
    suggestions = checker.suggest_cleaning_steps()
    assert "handle_outliers" in suggestions
    assert "Suggested Cleaning Steps" in capsys.readouterr().out


def test_default_zscore_threshold_is_three(rng):
    checker = ExploratoryDataChecker(pd.DataFrame({"x": rng.normal(size=2000)}))
    result = checker.check_outliers(method="zscore")["outliers_by_column"]["x"]
    assert result["threshold"] == 3.0
    assert result["percent_flagged"] < 0.01


def test_mad_uses_modified_z_score(rng):
    values = list(rng.normal(size=999)) + [4.5]
    result = ExploratoryDataChecker(pd.DataFrame({"x": values})).check_outliers(method="mad")
    assert result["outliers_by_column"]["x"]["outlier_range"][1] == 4.5


def test_mad_handles_zero_median_absolute_deviation():
    values = [0.0] * 80 + [1.0] * 10 + [100.0]
    result = ExploratoryDataChecker(pd.DataFrame({"x": values})).check_outliers(method="mad")
    assert result["outliers_by_column"]["x"]["count"] == 1


def test_unsupported_outlier_method_raises():
    with pytest.raises(ValueError):
        ExploratoryDataChecker(pd.DataFrame({"x": [1.0, 2.0]})).check_outliers(method="bogus")
