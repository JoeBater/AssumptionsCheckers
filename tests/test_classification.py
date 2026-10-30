import numpy as np
import pandas as pd
import pytest

from assumptions.ClassificationAssumptionsChecker import ClassificationAssumptionsChecker
from overlays.ClassificationOverlay import ClassificationOverlay


def _recommendation(checker, algorithm):
    return next(r for r in checker.report["recommendations"] if r["algorithm"] == algorithm)


def test_independent_features_do_not_penalise_logistic_regression(classification_df):
    checker = ClassificationAssumptionsChecker(classification_df, "t")
    checker.check_assumptions()
    lr = _recommendation(checker, "LogisticRegression")
    assert lr["suitability"] == "medium"
    assert "High multicollinearity detected" not in lr["notes"]


def test_separability_with_missing_rows_matches_complete_case(classification_df):
    with_missing = classification_df.copy()
    with_missing.loc[::4, "p"] = np.nan
    complete = with_missing.dropna().reset_index(drop=True)

    got = ClassificationOverlay(with_missing, "t", suppress_missing_warnings=True).check_separability()
    expected = ClassificationOverlay(complete, "t", suppress_missing_warnings=True).check_separability()
    assert got["overlap_score"] == expected["overlap_score"]


def test_checker_does_not_modify_caller_df(classification_df):
    df = classification_df.copy()
    df["tag"] = ["NA"] + ["ok"] * (len(df) - 1)
    original = df.copy()
    ClassificationAssumptionsChecker(df, "t")
    pd.testing.assert_frame_equal(df, original)


def test_collinear_features_flag_logistic_regression(classification_df, rng):
    df = classification_df.copy()
    df["p2"] = df["p"] * 2 + rng.normal(scale=0.01, size=len(df))
    checker = ClassificationAssumptionsChecker(df, "t")
    results = checker.check_assumptions()
    assert results["multicollinearity"]["has_multicollinearity"] is True
    assert _recommendation(checker, "LogisticRegression")["suitability"] == "low"


def test_skipped_multicollinearity_is_unknown_not_a_violation(classification_df):
    df = classification_df.copy()
    df.loc[:9, "q"] = np.nan  # a few missing values make the VIF check skip
    checker = ClassificationAssumptionsChecker(df, "t")
    results = checker.check_assumptions()
    assert results["multicollinearity"]["has_multicollinearity"] is None
    lr = _recommendation(checker, "LogisticRegression")
    assert lr["suitability"] == "medium"
    assert "Multicollinearity check skipped" in lr["notes"]


def test_no_algorithm_prints_note_and_recommends(classification_df, capsys):
    checker = ClassificationAssumptionsChecker(classification_df, "t")
    checker.check_assumptions()
    assert "No algorithm specified" in capsys.readouterr().out
    assert "recommendations" in checker.report


def test_logistic_regression_runs_targeted_checks(classification_df):
    checker = ClassificationAssumptionsChecker(classification_df, "t", algorithm="LogisticRegression")
    results = checker.check_assumptions()
    assert results["status"] == "suitable"
    assert results["scaling_issues"]["required_for_model"] is True
    assert {"multicollinearity", "separability", "redundancy"} <= results.keys()
    assert "recommendations" not in checker.report


@pytest.mark.parametrize("algorithm, required", [
    ("SVC", True), ("KNeighborsClassifier", True),
    ("DecisionTreeClassifier", False), ("RandomForestClassifier", False), ("GradientBoostingClassifier", False),
])
def test_each_algorithm_reports_status_and_scaling(classification_df, algorithm, required, capsys):
    checker = ClassificationAssumptionsChecker(classification_df, "t", algorithm=algorithm)
    results = checker.check_assumptions()
    checker.report_assumptions()
    assert results["status"] in {"suitable", "conditionally_permissible"}
    assert results["scaling_issues"]["required_for_model"] is required
    assert "Assumption Diagnostics Report" in capsys.readouterr().out


def test_unknown_algorithm_fails_softly(classification_df, capsys):
    checker = ClassificationAssumptionsChecker(classification_df, "t", algorithm="XGBClassifier")
    results = checker.check_assumptions()
    assert results["status"] == "unknown"
    assert "LogisticRegression" in capsys.readouterr().out  # help listing


def test_gradient_boosting_imbalance_advice_uses_sample_weight(classification_df):
    df = classification_df.copy()
    df["t"] = 0
    df.loc[:9, "t"] = 1  # 5% minority class
    checker = ClassificationAssumptionsChecker(df, "t", algorithm="GradientBoostingClassifier")
    reason = checker.check_assumptions()["reason"]
    assert "sample_weight" in reason
    assert "scale_pos_weight" not in reason


def test_skipped_separability_reports_complete_fraction(classification_df):
    df = classification_df.copy()
    df.loc[:119, "p"] = np.nan  # 60% of rows incomplete
    result = ClassificationOverlay(df, "t", suppress_missing_warnings=True).check_separability()
    assert result["separable"] is None
    assert "Only 40.0% of rows are complete" in result["notes"]


def test_vif_uses_an_intercept(rng):
    from overlays.SharedOverlay import SharedOverlay
    df = pd.DataFrame({"a": rng.normal(100, 1, 300), "b": rng.normal(50, 1, 300), "t": rng.integers(0, 2, 300)})
    result = SharedOverlay(df, "t").check_multicollinearity()
    assert result["has_multicollinearity"] is False
    assert all(r["VIF"] < 2 for r in result["details"])


def test_separability_is_scale_invariant(classification_df):
    rescaled = classification_df.copy()
    rescaled["q"] *= 1000
    original = ClassificationOverlay(classification_df.copy(), "t").check_separability()
    scaled = ClassificationOverlay(rescaled, "t").check_separability()
    assert scaled["overlap_score"] == pytest.approx(original["overlap_score"])


def test_separability_excludes_non_numeric_columns(classification_df, rng):
    df = classification_df.copy()
    df["colour"] = rng.choice(["red", "blue"], len(df))
    result = ClassificationOverlay(df, "t").check_separability()
    assert result["separable"] is not None
    assert result["excluded_columns"] == ["colour"]
