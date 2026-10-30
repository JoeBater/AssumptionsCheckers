import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

DEFAULT_MISSING_TOKENS = ["na", "NA", "NaN", "missing", "?", ".", "-999", "-9999"]


class MissingnessAwareOverlay:
    """Base for overlays whose checks run on the complete-case subset of the data."""

    def __init__(self, df, target, visualize=False, missingness_threshold=0.5,
                 suppress_missing_warnings=False, missing_tokens=None):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.missingness_threshold = missingness_threshold  # proportion of rows
        self.report = {}

        # Replace placeholder tokens (e.g. "?") with NaN; the checkers pass in their own copy
        if missing_tokens:
            self.df.replace(to_replace=set(missing_tokens), value=np.nan, inplace=True)

        self.cell_missing_frac = self.df.isnull().mean().mean()  # proportion of cells
        self.df_no_missing = self.df.dropna()
        self.row_missing_frac = (len(self.df) - len(self.df_no_missing)) / len(self.df) if len(self.df) else 0.0
        if self.row_missing_frac > 0:
            self.report["missing_data"] = (
                f"Warning: {self.row_missing_frac:.1%} of rows contain missing values. "
                + "\nTests will run on unmissing subset. Results may not generalize. "
                + "\nConsider imputation or removing rows/columns with excessive missingness."
            )
            # Only print if not suppressed (for when we handle missingness centrally)
            if not suppress_missing_warnings:
                print(self.report["missing_data"])

    @property
    def working_df(self):
        """The complete-case subset when missingness exists, otherwise the full frame."""
        return self.df_no_missing if self.row_missing_frac > 0 else self.df

    @property
    def data_used(self):
        return "Unmissing subset" if self.row_missing_frac > 0 else "Full dataset"

    def too_much_missing(self, check_name):
        """Record an error and return True if too many rows are incomplete to run check_name."""
        if self.row_missing_frac <= self.missingness_threshold:
            return False
        self.report["missing_data"] = (
            f"Error: {self.row_missing_frac:.1%} of rows have missing values, above the "
            f"{self.missingness_threshold:.0%} threshold. \n{check_name} cannot proceed."
        )
        return True


def split_numeric(X):
    """Return (numeric columns of X, names of excluded columns); booleans count as non-numeric."""
    excluded = [col for col in X.columns
                if not pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])]
    return X.drop(columns=excluded), excluded


def _vif_table(X):
    # variance_inflation_factor regresses on the other exog columns as given, so it needs an
    # explicit intercept; without one, any feature with a non-zero mean gets a huge VIF.
    exog = sm.add_constant(X, has_constant="add").values
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vifs = []
        for i in range(1, exog.shape[1]):  # column 0 is the intercept
            try:
                vifs.append(variance_inflation_factor(exog, i))
            except (np.linalg.LinAlgError, ValueError):
                vifs.append(float("inf"))
    return pd.DataFrame({"feature": X.columns.tolist(), "VIF": pd.to_numeric(vifs, errors="coerce")})


def vif_drop_set(X, threshold=10):
    """Compute VIFs, greedily dropping the highest until none exceeds threshold.

    Returns the final VIF table and the list of dropped features.
    """
    vif_data = _vif_table(X)
    too_high = vif_data[vif_data["VIF"] > threshold]

    drop_set = []
    remaining = X
    while not too_high.empty and remaining.shape[1] > 1:
        to_drop = too_high.loc[too_high["VIF"].idxmax(), "feature"]
        drop_set.append(to_drop)
        remaining = remaining.drop(columns=[to_drop])
        vif_data = _vif_table(remaining)
        too_high = vif_data[vif_data["VIF"] > threshold]

    return vif_data, drop_set
