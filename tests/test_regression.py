import pandas as pd
import pytest

from assumptions.RegressionAssumptionsChecker import RegressionAssumptionsChecker


def test_categorical_column_is_excluded_from_ols_checks(regression_df, rng):
    df = regression_df.copy()
    df["cat"] = rng.choice(["u", "v"], len(df))
    checker = RegressionAssumptionsChecker(df, "t", algorithm="LinearRegression")
    results = checker.check_assumptions()
    assert results["linearity"]["linearity"] is not None
    assert "cat" in results["linearity"]["notes"]


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


@pytest.mark.parametrize("algorithm, required", [
    ("LinearRegression", False), ("Ridge", True), ("Lasso", True), ("ElasticNet", True),
])
def test_scaling_requirement_per_linear_model(regression_df, algorithm, required):
    results = RegressionAssumptionsChecker(regression_df, "t", algorithm=algorithm).check_assumptions()
    assert results["scaling"]["required_for_model"] is required


def test_heteroscedasticity_is_conditional_not_unsuitable(regression_df, rng):
    df = regression_df.copy()
    df["t"] = df["p"] + rng.normal(size=len(df)) * (0.1 + df["q"] - df["q"].min())  # variance grows with q
    results = RegressionAssumptionsChecker(df, "t", algorithm="LinearRegression").check_assumptions()
    assert not results["homoscedasticity"]["homoscedasticity"]
    assert results["status"] == "conditionally_permissible"
    assert "robust" in results["reason"]


@pytest.fixture
def mostly_missing_df(regression_df):
    df = regression_df.copy()
    df.loc[:119, "p"] = float("nan")  # 60% of rows incomplete, above the 0.5 threshold
    return df


def test_skipped_checks_are_unknown_not_violations(mostly_missing_df, capsys):
    checker = RegressionAssumptionsChecker(mostly_missing_df, "t", algorithm="LinearRegression")
    results = checker.check_assumptions()
    assert results["status"] == "unknown"
    checker.report_assumptions()
    assert "homoscedasticity: skipped" in capsys.readouterr().out


def test_skipped_checks_do_not_penalise_recommendations(mostly_missing_df):
    checker = RegressionAssumptionsChecker(mostly_missing_df, "t")
    results = checker.check_assumptions()
    assert results["multicollinearity"]["has_multicollinearity"] is None
    lr = next(r for r in checker.report["recommendations"] if r["algorithm"] == "LinearRegression")
    assert lr["suitability"] == "medium"


def test_missing_percentage_is_formatted_as_percent(mostly_missing_df):
    checker = RegressionAssumptionsChecker(mostly_missing_df, "t")
    assert "60.0% of rows contain missing values" in checker.regression_overlay.report["missing_data"]


def test_no_algorithm_prints_note(regression_df, capsys):
    RegressionAssumptionsChecker(regression_df, "t").check_assumptions()
    assert "No algorithm specified" in capsys.readouterr().out
