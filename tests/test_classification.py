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
