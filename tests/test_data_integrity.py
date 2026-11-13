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


def test_embedded_rows_by_regex_flags_delimited_cells():
    df = pd.DataFrame({"s": ["plain", "a|b|c|d", None], "n": [1, 2, 3]})
    result = DataIntegrityOverlay(df).check_embedded_rows_by_regex()
    assert [(idx, col) for idx, col, _ in result["embedded_cells"]] == [(1, "s")]


def test_run_checks_returns_results(messy_df):
    results = DataIntegrityOverlay(messy_df.copy()).run_checks()
    assert {"missingness", "whitespace", "constancy", "cardinality", "duplicates", "embedded_rows"} <= results.keys()


def test_string_dtype_columns_are_checked():
    df = pd.DataFrame({"id": pd.array([f"u{i}" for i in range(20)], dtype="string"),
                       "s": pd.array(["a|b|c|d"] + ["x"] * 19, dtype="string")})
    overlay = DataIntegrityOverlay(df)
    assert "id" in overlay.check_high_cardinality()["high_cardinality"]
    assert len(overlay.check_embedded_rows_by_regex()["embedded_cells"]) == 1


def test_numeric_sentinels_are_detected():
    df = pd.DataFrame({"a": [1, -999, 3], "b": [1.0, 2.0, -999.0], "c": ["x", "-999", "y"]})
    hits = DataIntegrityOverlay(df).check_missingness()["custom_missing_hits"]
    assert hits == {"a": 1, "b": 1, "c": 1}


def test_high_missing_rows_with_string_index():
    df = pd.DataFrame({c: [np.nan, 1, 2] for c in "abcde"}, index=["r1", "r2", "r3"])
    assert DataIntegrityOverlay(df).check_missingness()["high_missing_rows"] == {"r1": 1.0}
