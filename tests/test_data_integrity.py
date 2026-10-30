import numpy as np
import pandas as pd
import pytest

from assumptions.DataIntegrityChecker import DataIntegrityChecker
from overlays.DataIntegrityOverlay import DataIntegrityOverlay


@pytest.fixture
def messy_df():
    return pd.DataFrame({
        "a": [1, 2, np.nan, 4, 5, 6],
        "b": ["x", " y", None, "z", "w", "v"],
        "id": ["u1", "u2", "u3", "u4", "u5", "u6"],
    })


def test_missingness_without_high_missing_columns_does_not_crash(messy_df):
    result = DataIntegrityOverlay(messy_df).check_missingness()
    assert result["status"] == "ok"


def test_whitespace_fix_preserves_missing_values(messy_df):
    overlay = DataIntegrityOverlay(messy_df.copy())
    overlay.check_whitespace()
    assert overlay.df["b"].tolist()[:2] == ["x", "y"]
    assert overlay.df["b"].isna().sum() == 1


def test_checker_does_not_modify_caller_df(messy_df):
    original = messy_df.copy()
    DataIntegrityChecker(messy_df).run_all_checks()
    pd.testing.assert_frame_equal(messy_df, original)


def test_report_includes_cardinality_and_duplicate_details():
    df = pd.DataFrame({
        "id": [f"u{i}" for i in range(20)] + ["u0"],
        "v": list(range(20)) + [0],
    })
    checker = DataIntegrityChecker(df)
    checker.run_all_checks()
    report = checker.report()
    assert "`id` (" in report  # high-cardinality column details
    assert "Preview: indices" in report  # duplicate row details
