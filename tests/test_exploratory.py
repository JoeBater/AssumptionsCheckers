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
