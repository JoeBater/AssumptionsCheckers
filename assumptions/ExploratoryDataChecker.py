import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, zscore
from statsmodels import robust

class ExploratoryDataChecker:
    def __init__(self, df, visualize=False, replace_missing=False):
        self.df = df.copy()
        self.replace_missing = replace_missing
        self.visualize = visualize
        self.report = {}

    def check_missing_values(self, custom_missing=None, replace_missing=False):
        """
        Checks for missing values, including custom placeholders.

        Parameters:
        - custom_missing: list of strings/numbers treated as missing
        - replace_missing: bool, whether to replace custom values with np.nan

        Returns:
        - dict with missing value summary and advisory notes
        """
        if custom_missing is None:
            custom_missing = ["?", "missing", "-9999", "-999", "NA", "N/A"]

        df = self.df.copy()
        if replace_missing:
            df = df.replace(custom_missing, np.nan)

        # Standard missing values
        standard_missing = df.isnull().sum()

        # Custom missing values per column
        custom_missing_by_column = {
            col: sum((self.df[col] == val).sum() for val in custom_missing)
            for col in self.df.columns
        }

        # Filter to only columns with actual missing values
        column_summary = {}
        for col in self.df.columns:
            std_missing = int(standard_missing[col])
            custom_missing = int(custom_missing_by_column[col])
            total = std_missing + custom_missing
            if total > 0:
                column_summary[col] = {
                    "standard_missing": std_missing,
                    "custom_missing": custom_missing,
                    "total": total
                }

        total_missing = sum(v["total"] for v in column_summary.values())

        notes = []
        if total_missing == 0:
            notes.append("No missing values detected.")
        else:
            if any(v["custom_missing"] > 0 for v in column_summary.values()) and not replace_missing:
                notes.append("Custom missing values detected but not replaced. Consider handling manually.")
            if any(v["standard_missing"] > 0 for v in column_summary.values()):
                notes.append("Standard NaNs detected.")

        return {
            "total_missing": total_missing,
            "missing_summary": column_summary,
            "custom_missing_values": custom_missing,
            "replace_missing": replace_missing,
            "notes": notes
        }



    

    def check_outliers(self, method="auto", threshold=1.5):
        """
        Detects outliers using IQR, Z-score, MAD, or auto-selected method per column.

        Parameters:
        - method: "auto", "iqr", "zscore", "mad"
        - threshold: float, sensitivity parameter

        Returns:
        - dict with outlier counts, values, methods used, and notes
        """
        numeric = self.df.select_dtypes(include=[np.number])
        outlier_summary = {}

        for col in numeric.columns:
            x = numeric[col].dropna()

            # Auto-select method based on distribution
            if method == "auto":
                skew_val = skew(x)
                kurt_val = kurtosis(x)
                if abs(skew_val) < 1 and kurt_val < 3:
                    chosen = "zscore"
                elif abs(skew_val) < 2 and kurt_val < 5:
                    chosen = "iqr"
                else:
                    chosen = "mad"
            else:
                chosen = method

            # Apply chosen method
            if chosen == "iqr":
                q1, q3 = x.quantile(0.25), x.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
                mask = (x < lower) | (x > upper)

            elif chosen == "zscore":
                zs = zscore(x)
                mask = np.abs(zs) > threshold

            elif chosen == "mad":
                median = x.median()
                mad = robust.mad(x)
                mod_z = 0.6745 * (x - median) / mad
                mask = np.abs(mod_z) > threshold

            else:
                raise ValueError(f"Unsupported method: {chosen}")

            flagged = x[mask].tolist()
            inliers = x[~mask].tolist()
            outlier_summary[col] = {
                "method": chosen,
                "count": len(flagged),
                "percent_flagged": round(len(flagged) / len(x), 4),
                "outlier_range": [min(flagged), max(flagged)] if len(flagged) > 0 else None,
                "inlier_range": [min(inliers), max(inliers)] if len(inliers) > 0 else None,
            }

        sorted_outliers = dict(
            sorted(outlier_summary.items(), key=lambda item: item[1]["percent_flagged"], reverse=True)
        )

        total_outliers = sum(v["count"] for v in outlier_summary.values())
        notes = (
            "Outlier detection method auto-selected per column based on distribution shape."
            if method == "auto" else
            f"Outlier detection applied using {method.upper()} method across all numeric columns."
        )
        

        return {
            "outliers_by_column": sorted_outliers,
            "total_outliers": total_outliers,
            "method": method,
            "threshold": threshold,
            "notes": notes
        }


    def check_whitespace_strings(self):
        string_cols = self.df.select_dtypes(include="object")
        flagged = {}
        for col in string_cols.columns:
            mask = string_cols[col].astype(str).apply(lambda x: x != x.strip())
            flagged[col] = int(mask.sum())
        return {
            "whitespace_issues": flagged,
            "notes": "Leading/trailing whitespace detected in string columns."
        }

    def check_constant_columns(self):
        constant_cols = [col for col in self.df.columns if self.df[col].nunique(dropna=False) <= 1]
        return {
            "constant_columns": constant_cols,
            "notes": f"{len(constant_cols)} constant columns detected."
        }

    def check_high_cardinality(self, threshold=50):
        categorical = self.df.select_dtypes(include="object")
        high_card = {col: self.df[col].nunique() for col in categorical.columns if self.df[col].nunique() > threshold}
        return {
            "high_cardinality": high_card,
            "threshold": threshold,
            "notes": f"{len(high_card)} columns exceed cardinality threshold."
        }

    def check_overlong_inputs(self):
        overlong = pd.DataFrame()
        for column in self.df.columns:
            median_length = np.median(len(self.df[column]))
            overlong = pd.concat([overlong, self.df.filter(len(self.df[column]) > median_length)])
        return {
            "rows with overlong elements": overlong
        }

    def run_all_checks(self):
        self.report["missing_values"] = self.check_missing_values()
        self.report["outliers"] = self.check_outliers()
        self.report["whitespace"] = self.check_whitespace_strings()
        self.report["constant_columns"] = self.check_constant_columns()
        self.report["high_cardinality"] = self.check_high_cardinality()
        #self.report["overlong"] = self.check_overlong_inputs()
        
        self.present_exploration()


    def suggest_cleaning_steps(self):
        """
        Suggests cleaning code snippets based on detected issues.
        Returns a dictionary of actionable code suggestions.
        """
        suggestions = {}

        # Missing values
        mv = self.report.get("missing_values", {})
        if mv.get("total_missing", 0) > 0:
            suggestions["handle_missing_values"] = [
                "# Replace custom missing values with NaN",
                f"df.replace({mv['custom_missing_values']}, np.nan, inplace=True)",
                "# Optionally drop rows or columns with too many NaNs",
                "df.dropna(axis=0, thresh=MIN_NON_NAN_VALUES)  # or axis=1"
            ]

        # Outliers
        outliers = self.report.get("outliers", {}).get("outliers_by_column", {})
        flagged_outliers = [col for col, count in outliers.items() if count > 0]
        if flagged_outliers:
            suggestions["handle_outliers"] = [
                "# Example: cap or remove outliers using IQR method",
                "for col in ['" + "', '".join(flagged_outliers) + "']:",
                "    Q1 = df[col].quantile(0.25)",
                "    Q3 = df[col].quantile(0.75)",
                "    IQR = Q3 - Q1",
                "    lower = Q1 - 1.5 * IQR",
                "    upper = Q3 + 1.5 * IQR",
                "    df = df[(df[col] >= lower) & (df[col] <= upper)]"
            ]

        # Whitespace issues
        ws = self.report.get("whitespace", {}).get("whitespace_issues", {})
        flagged_ws = [col for col, count in ws.items() if count > 0]
        if flagged_ws:
            suggestions["strip_whitespace"] = [
                "# Strip leading/trailing whitespace from string columns",
                "for col in ['" + "', '".join(flagged_ws) + "']:",
                "    df[col] = df[col].astype(str).str.strip()"
            ]

        # Constant columns
        constants = self.report.get("constant_columns", {}).get("constant_columns", [])
        if constants:
            suggestions["drop_constant_columns"] = [
                "# Drop constant columns",
                f"df.drop(columns={constants}, inplace=True)"
            ]

        # High cardinality
        high_card = self.report.get("high_cardinality", {}).get("high_cardinality", {})
        if high_card:
            suggestions["handle_high_cardinality"] = [
                "# Consider encoding or reducing high-cardinality categorical features",
                "from sklearn.preprocessing import OrdinalEncoder",
                "encoder = OrdinalEncoder()",
                f"df[list({list(high_card.keys())})] = encoder.fit_transform(df[list({list(high_card.keys())})])"
            ]

        return suggestions

    def present_exploration(self):
        print("\n📊 Exploratory Data Diagnostics")
        print("=" * 40)

        if not self.report:
            print("⚠️ No diagnostics found. Run `run_all_checks()` first.")
            return

        for section, result in self.report.items():
            title = section.replace("_", " ").title()
            print(f"\n🔍 {title}")
            print("-" * 30)

            if isinstance(result, dict):
                for key, value in result.items():
                    if isinstance(value, dict):
                        print(f"{key}:")
                        for subkey, subval in value.items():
                            print(f"  {subkey}: {subval}")
                    elif isinstance(value, list):
                        print(f"{key}: {', '.join(map(str, value))}")
                    else:
                        print(f"{key}: {value}")
            else:
                print(result)

        # Cleaning suggestions
        suggestions = self.suggest_cleaning_steps()
        if suggestions:
            print("\n🧼 Suggested Cleaning Steps")
            print("=" * 40)
            for section, lines in suggestions.items():
                title = section.replace("_", " ").title()
                print(f"\n🔧 {title}")
                print("-" * 30)
                for line in lines:
                    print(f"  {line}")
        else:
            print("\n✅ No cleaning suggestions needed — data looks structurally sound.")
