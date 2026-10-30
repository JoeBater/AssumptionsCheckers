NO_ALGORITHM_NOTE = (
    "ℹ️ No algorithm specified — running all checks and ranking candidate models. "
    "Pass algorithm='...' to check a specific model (see help_algorithms())."
)

# Result keys holding a check's verdict: True/False, or None when the check was skipped
VERDICT_KEYS = {
    "linearity", "homoscedasticity", "normality", "has_multicollinearity",
    "scaling_needed", "tree_suitability", "separable", "is_imbalanced",
}


SUITABILITY_ORDER = {"high": 3, "medium": 2, "low": 1, "unsuitable": 0}


def verdict(result, key):
    """Return a check's True/False verdict, or None if the check was skipped."""
    value = result.get(key) if isinstance(result, dict) else None
    # Overlays often return numpy bools, which fail identity checks like `is False`
    return None if value is None else bool(value)


def print_check_result(result):
    """Print one check's result dict, showing skipped verdicts as 'skipped'."""
    for subkey, value in result.items():
        # Skip redundant missingness messages from individual checks
        if subkey in ["notes", "recommendation"] and "missing" in str(value).lower():
            continue
        if value is None:
            if subkey in VERDICT_KEYS:
                print(f"{subkey}: skipped")
            continue
        print(f"{subkey}: {value}")


class BaseAssumptionsChecker:
    """Shared plumbing for the model-family checkers: missingness summary, ranking and reporting."""

    def __init__(self, df, target, algorithm=None):
        # Overlays normalise missing-value tokens in place, so never touch the caller's frame
        self.df = df.copy()
        self.target = target
        self.algorithm = algorithm
        self.results = {}
        self.report = {}

    def _generate_missingness_analysis(self):
        """Generate a comprehensive missingness analysis with recommendations."""
        missing_info = self.df.isnull()
        total_missing = missing_info.sum().sum()
        total_cells = len(self.df) * len(self.df.columns)
        overall_missing_pct = (total_missing / total_cells) * 100

        if total_missing == 0:
            return {
                "has_missing": False,
                "overall_pct": 0.0,
                "summary": "✅ No missing values detected in the dataset.",
                "recommendation": None
            }

        # Analyze missing patterns
        rows_with_missing = missing_info.any(axis=1).sum()
        rows_missing_pct = (rows_with_missing / len(self.df)) * 100

        # Column-wise analysis
        cols_missing = missing_info.sum()
        cols_with_missing = cols_missing[cols_missing > 0].sort_values(ascending=False)
        cols_missing_pct = (cols_with_missing / len(self.df)) * 100

        # Most affected columns (top 5)
        most_affected = cols_missing_pct.head(5)

        # Generate recommendation based on missingness patterns
        recommendation = self._get_missingness_recommendation(overall_missing_pct, rows_missing_pct, cols_missing_pct)

        # Create summary message
        summary = (
            f"⚠️ Missing data detected: {overall_missing_pct:.1f}% of all values missing. "
            f"{rows_missing_pct:.1f}% of rows affected. "
            f"Most affected columns: {', '.join([f'{col} ({pct:.1f}%)' for col, pct in most_affected.head(3).items()])}."
        )

        return {
            "has_missing": True,
            "overall_pct": round(overall_missing_pct, 2),
            "rows_affected_pct": round(rows_missing_pct, 2),
            "most_affected_columns": most_affected.round(1).to_dict(),
            "summary": summary,
            "recommendation": recommendation
        }

    def _get_missingness_recommendation(self, overall_pct, rows_pct, cols_pct):
        """Generate missingness handling recommendations based on patterns."""
        if overall_pct < 5:
            return "Low missingness: Consider listwise deletion or simple imputation."
        elif overall_pct < 15:
            if any(cols_pct > 50):
                return "Moderate missingness with some heavily affected columns: Consider dropping high-missing columns, then impute remaining."
            else:
                return "Moderate missingness: Consider multiple imputation or robust models (XGBoost, CatBoost)."
        elif overall_pct < 30:
            return "High missingness: Strongly recommend robust models (XGBoost, CatBoost) or sophisticated imputation techniques."
        else:
            return "Very high missingness: Consider data collection improvement, advanced imputation, or models specifically designed for sparse data."

    def _rank_recommendations(self, recommendations):
        """Sort by suitability (high > medium > low > unsuitable) and store in the report."""
        recommendations.sort(key=lambda x: SUITABILITY_ORDER.get(x["suitability"], 0), reverse=True)
        self.report["recommendations"] = recommendations
        return recommendations

    def export_report(self, format="dict"):
        if format == "json":
            import json
            return json.dumps(self.report, indent=2, default=str)
        return self.report

    def report_assumptions(self):
        print("\n🔍 Assumption Diagnostics Report")
        print("=" * 40)

        # Show consolidated missingness analysis first
        if self.missingness_analysis["has_missing"]:
            print(f"\n📊 Data Quality Summary")
            print("-" * 30)
            print(self.missingness_analysis["summary"])
            if self.missingness_analysis["recommendation"]:
                print(f"💡 Recommendation: {self.missingness_analysis['recommendation']}")

            # Show detailed column breakdown if requested
            if len(self.missingness_analysis["most_affected_columns"]) > 3:
                print(f"\nDetailed missing data by column:")
                for col, pct in list(self.missingness_analysis["most_affected_columns"].items())[:5]:
                    print(f"  • {col}: {pct}%")

        assumptions = self.report.get("assumptions", self.results)

        for key, result in assumptions.items():
            print(f"\n🧠 {key.replace('_', ' ').title()}")
            print("-" * 30)

            if isinstance(result, dict):
                print_check_result(result)
            else:
                print(result)

        if "recommendations" in self.report:
            print("\n⚠️ Model Recommendations (ordered by suitability)")
            print("-" * 30)
            for rec in self.report["recommendations"]:
                print(f"\n• {rec['algorithm']} [{rec['suitability'].upper()}]")
                print(f"  Reason: {rec['reason']}")
                if rec.get('notes'):
                    for note in rec['notes']:
                        print(f"  - {note}")
