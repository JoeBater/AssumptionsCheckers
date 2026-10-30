import pandas as pd
import pytest

from assumptions.RegressionAssumptionsChecker import RegressionAssumptionsChecker


@pytest.mark.xfail(strict=True, reason="bug #7: OLS checks fail on non-numeric columns")
def test_categorical_column_is_excluded_from_ols_checks(regression_df, rng):
    df = regression_df.copy()
    df["cat"] = rng.choice(["u", "v"], len(df))
    checker = RegressionAssumptionsChecker(df, "t", algorithm="LinearRegression")
    results = checker.check_assumptions()
    assert results["linearity"]["linearity"] is not None
    assert "cat" in results["linearity"]["notes"]


@pytest.mark.xfail(strict=True, reason="bug #6: missing tokens replaced in caller df")
def test_checker_does_not_modify_caller_df(regression_df):
    df = regression_df.copy()
    df["tag"] = ["NA"] + ["ok"] * (len(df) - 1)
    original = df.copy()
    RegressionAssumptionsChecker(df, "t")
    pd.testing.assert_frame_equal(df, original)


@pytest.mark.parametrize("algorithm", [
    None, "LinearRegression", "KNeighborsRegressor", "RandomForestRegressor",
])
def test_end_to_end_smoke(regression_df, algorithm, capsys):
    checker = RegressionAssumptionsChecker(regression_df, "t", algorithm=algorithm)
    checker.check_assumptions()
    checker.report_assumptions()
    assert "Assumption Diagnostics Report" in capsys.readouterr().out
