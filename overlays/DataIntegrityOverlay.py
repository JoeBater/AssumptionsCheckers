import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import re
import warnings
from collections import Counter
from scipy.stats import shapiro
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns

import warnings

class DataIntegrityOverlay:

    FIX_WHITESPACE = True  # Default behavior: auto-fix whitespace issues
    EMBEDDED_BY_LENGTH = False   # Default behaviour: detect possible embedded data by median data length

    def __init__(self, df, custom_missing_values=None, visualize=False):
        self.df = df
        self.custom_missing_values=custom_missing_values
        self.visualize = visualize

    def run_checks(self):
        results = {}

        results["missingness"] = self.check_missingness()
        results["whitespace"] = self.check_whitespace()
        results["constancy"] = self.check_constancy()
        results["cardinality"] = self.check_high_cardinality()
        results["duplicates"] = self.check_duplicate_rows()

        if self.EMBEDDED_BY_LENGTH:
            results["embedded_rows"] = self.check_embedded_rows_by_length()
        else:
            results["embedded_rows"] = self.check_embedded_rows_by_regex()

    # =======================================
    # Individual checks
    # =======================================

    # Missingness patterns (MCAR/MAR/MNAR hints)
    def check_missingness(self, custom_missing_values=None, col_threshold=0.5, row_threshold=0.8):
        default_missing_values = ["missing", ".", "?", "-999", "-9999"]
        missing_values = set(custom_missing_values) if custom_missing_values else set(default_missing_values)

        df = self.df.copy()
        missing_mask = df.isna()

        # Include custom proxies
        for col in df.columns:
            proxy_mask = df[col].astype(str).str.strip().isin(missing_values)
            missing_mask[col] = missing_mask[col] | proxy_mask

        # Column and row missingness
        col_missing_pct = missing_mask.mean()
        high_missing_cols = col_missing_pct[col_missing_pct > col_threshold].sort_values(ascending=False)
        row_missing_pct = missing_mask.mean(axis=1)
        high_missing_rows = row_missing_pct[row_missing_pct > row_threshold].sort_values(ascending=False)

        # MCAR/MAR hints with safe correlation
        hints = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for col in df.columns:
                if missing_mask[col].sum() > 0:
                    correlations = {}
                    for other_col in df.select_dtypes(include=[np.number]).columns:
                        if other_col != col and df[other_col].std() > 0:
                            try:
                                corr = missing_mask[col].astype(int).corr(df[other_col])
                                if abs(corr) > 0.2:
                                    correlations[other_col] = round(corr, 3)
                            except Exception:
                                continue
                    hints[col] = {
                        "pattern": "Likely MAR" if correlations else "Possibly MCAR",
                        "correlated_with": correlations
                    }

        summary = (
            f"Missingness detected in {len(high_missing_cols)} columns and {len(high_missing_rows)} rows."
            if high_missing_cols.any() or high_missing_rows
            else "No significant missingness found."
        )

        status = "ok" if not high_missing_cols.any() and not high_missing_rows else "warning"

        actions = []

        if not high_missing_cols.empty:
            actions.append("# Consider imputing or dropping columns with high missingness")
            for col, pct in high_missing_cols.items():
                actions.append(f"# Column `{col}` has {pct*100:.1f}% missing")
                actions.append(f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())  # or use domain-specific imputation")

        if not high_missing_rows.empty:
            actions.append("# Consider dropping rows with excessive missingness")
            actions.append(f"df.drop(index={list(high_missing_rows.keys())}, inplace=True)")


        return {
            "high_missing_columns": high_missing_cols.round(3).to_dict(),
            "high_missing_rows": {int(idx): round(pct, 3) for idx, pct in row_missing_pct[row_missing_pct > row_threshold].items()},
            "custom_missing_hits": {col: int(missing_mask[col].sum()) for col in df.columns if missing_mask[col].sum() > 0},
            "missingness_hints": hints,
            "total_missing_cells": int(missing_mask.sum().sum()),
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }

    # Whitespace-only, null-like, and malformed entries
    def check_whitespace(self):
        df = self.df.copy()
        whitespace_issues = {}

        null_like = {"", " ", "\t", "\n", "n/a", "na", "none", "null", "?"}

        for col in df.columns:
            if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
                raw = df[col].astype(str)
                stripped = raw.str.strip()

                count_null_like = stripped.isin(null_like).sum()
                count_leading = (raw != raw.str.lstrip()).sum()
                count_trailing = (raw != raw.str.rstrip()).sum()

                total_issues = count_null_like + count_leading + count_trailing
                if total_issues > 0:
                    whitespace_issues[col] = {
                        "null_like": int(count_null_like),
                        "leading_space": int(count_leading),
                        "trailing_space": int(count_trailing),
                        "total": total_issues
                    }

                    if self.FIX_WHITESPACE:
                        self.df[col] = stripped  # Clean in-place

        actions = []

        if whitespace_issues and not self.FIX_WHITESPACE:
            actions.append("# Strip leading/trailing whitespace from affected columns")
            for col in whitespace_issues:
                actions.append(f"df['{col}'] = df['{col}'].astype(str).str.strip()")


        if whitespace_issues:
            if self.FIX_WHITESPACE:
                summary = f"Whitespace issues detected and fixed in {len(whitespace_issues)} columns."
            else:
                summary = f"Whitespace issues detected in {len(whitespace_issues)} columns."
            status = "warning"
        else:
            summary = "No whitespace issues found."
            status = "ok"


        return {
            "whitespace_issues": whitespace_issues,
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }

    # Constant columns and near-constant detection
    def check_constancy(self, threshold=0.99):
        df = self.df.copy()
        constant_columns = []
        near_constant_columns = {}

        for col in df.columns:
            unique_vals = df[col].nunique(dropna=True)
            total_vals = df[col].count()
            if unique_vals == 1:
                constant_columns.append(col)
            elif unique_vals > 1:
                top_freq = df[col].value_counts(dropna=True).iloc[0]
                top_ratio = top_freq / total_vals
                if top_ratio >= threshold:
                    near_constant_columns[col] = round(top_ratio, 3)

        summary = (
            f"{len(constant_columns)} constant and {len(near_constant_columns)} near-constant columns detected."
            if constant_columns or near_constant_columns
            else "No constant or near-constant columns found."
        )
        status = "ok" if not constant_columns and not near_constant_columns else "warning"

        actions = []
        if constant_columns:
            actions.append("# Drop constant columns")
            actions.append(f"df.drop(columns={constant_columns}, inplace=True)")

        if near_constant_columns:
            actions.append("# Consider dropping or encoding near-constant columns")
            for col, ratio in near_constant_columns.items():
                actions.append(f"# Column `{col}` has {ratio*100:.1f}% of a single value")

        return {
            "constant_columns": constant_columns,
            "near_constant_columns": near_constant_columns,
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }



    # High cardinality and ID-like column detection
    def check_high_cardinality(self, min_ratio=0.5, min_unique_ratio=0.05):
        df = self.df.copy()
        high_cardinality = {}

        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == "string":
                unique_count = df[col].nunique(dropna=True)
                total_count = df[col].count()
                unique_ratio = unique_count / total_count if total_count > 0 else 0
                dataset_ratio = unique_count / len(df)

                if unique_ratio > min_ratio and dataset_ratio > min_unique_ratio:
                    high_cardinality[col] = {
                        "unique_values": int(unique_count),
                        "total_values": int(total_count),
                        "unique_ratio": round(unique_ratio, 3),
                        "dataset_ratio": round(dataset_ratio, 3)
                    }

        # ✅ Summary and status
        if high_cardinality:
            summary = f"{len(high_cardinality)} high-cardinality columns detected."
            status = "warning"
        else:
            summary = "No high-cardinality columns found."
            status = "ok"

        # ✅ Recommended actions
        actions = []
        if high_cardinality:
            actions.append("# Consider encoding or dropping high-cardinality categorical columns")
            for col, stats in high_cardinality.items():
                actions.append(f"# Column `{col}` has {stats['unique_values']} unique values out of {stats['total_values']} entries")
                actions.append(f"# Example: df['{col}'] = df['{col}'].astype('category') or use hashing/embedding")

        return {
            "high_cardinality": high_cardinality,
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }


    # Duplicate rows and index integrity
    def check_duplicate_rows(self):
        df = self.df.copy()
        duplicate_mask = df.duplicated(keep=False)
        duplicate_rows = df[duplicate_mask]

        num_duplicates = duplicate_mask.sum()
        duplicate_indices = df.index[duplicate_mask].tolist()

        summary = (
            f"{num_duplicates} duplicated rows detected."
            if num_duplicates > 0 else "No duplicated rows found."
        )
        status = "warning" if num_duplicates > 0 else "ok"

        actions = []
        if num_duplicates > 0:
            actions.append("# Consider dropping duplicated rows")
            actions.append(f"df.drop_duplicates(inplace=True)")

        return {
            "duplicate_count": int(num_duplicates),
            "duplicate_indices": duplicate_indices[:10],  # preview
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }


    # Actionable cleaning suggestions with severity flags
    def recommended_actions(self):
        pass


    def check_embedded_rows_by_length(self, delimiter="|", min_fields=3, deviation_factor=2.0):
        df = self.df.copy()
        embedded_cells = []

        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == "string":
                lengths = df[col].dropna().astype(str).apply(len)
                median_len = lengths.median()
                threshold = median_len * deviation_factor

                for idx, val in df[col].dropna().items():
                    if isinstance(val, str) and len(val) > threshold and val.count(delimiter) >= min_fields:
                        embedded_cells.append((idx, col, val))

        # summary, status, actions as before...
        summary = (
            f"{len(embedded_cells)} embedded row-like cells detected."
            if embedded_cells else "No embedded row-like cells found."
        )
        status = "warning" if embedded_cells else "ok"

        actions = []
        if embedded_cells:
            actions.append("# Consider extracting embedded rows from pipe-delimited cells")
            for idx, col, val in embedded_cells[:5]:
                actions.append(f"# Row {idx}, column `{col}` contains embedded data: {val[:80]}...")

        return {
            "embedded_cells": embedded_cells,
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }
    
    

    def check_embedded_rows_by_regex(self, delimiter="|", sample_size=100):
        df = self.df.copy()
        embedded_cells = []

        for col in df.columns:
            if df[col].dtype == object or df[col].dtype.name == "string":
                samples = df[col].dropna().astype(str).sample(min(sample_size, len(df)))
                pattern = re.compile(r"^" + re.escape(samples.iloc[0]) + r"$")

                for val in samples:
                    pattern = re.compile(r"^" + re.escape(val) + r"$")  # simplistic fallback

                for idx, val in df[col].dropna().items():
                    if not pattern.match(str(val)) and delimiter in str(val):
                        embedded_cells.append((idx, col, val))

        # summary, status, actions as before...
        summary = (
            f"{len(embedded_cells)} embedded row-like cells detected."
            if embedded_cells else "No embedded row-like cells found."
        )
        status = "warning" if embedded_cells else "ok"

        actions = []
        if embedded_cells:
            actions.append("# Consider extracting embedded rows from pipe-delimited cells")
            for idx, col, val in embedded_cells[:5]:
                actions.append(f"# Row {idx}, column `{col}` contains embedded data: {val[:80]}...")

        return {
            "embedded_cells": embedded_cells,
            "report": summary,
            "status": status,
            "recommended_actions": actions
        }