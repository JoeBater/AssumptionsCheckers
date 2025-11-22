import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


from overlays.SharedOverlay import SharedOverlay
from overlays.ClassificationOverlay import ClassificationOverlay

class ClassificationAssumptionsChecker:
    def __init__(self, df, target, algorithm=None, visualize=False):
        self.df = df
        self.target = target
        self.algorithm = algorithm
        
        # Suppress individual overlay missingness warnings by setting to False temporarily
        # We'll generate a consolidated missingness report instead
        self.overlay = SharedOverlay(df, target, visualize, suppress_missing_warnings=True)
        self.classification_overlay = ClassificationOverlay(df, target, visualize, suppress_missing_warnings=True)
        self.report = {}
        
        # Generate consolidated missingness analysis
        self.missingness_analysis = self._generate_missingness_analysis()
        

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

    def check_assumptions(self):
        self.assumption_results = {
            "multicollinearity": self.overlay.check_multicollinearity(),
            "class_imbalance": self.overlay.check_class_imbalance(),
            "scaling_issues": self.overlay.check_scaling(),
            "redundancy": self.classification_overlay.check_redundancy(),
            "separability": self.classification_overlay.check_separability(),
        }
        self.recommend_models()
        return self.assumption_results

    def recommend_models(self):
        cautions = []
        if self.assumption_results.get("multicollinearity", False):
            cautions.append("Avoid LogisticRegression without regularization due to multicollinearity.")
        if self.assumption_results.get("class_imbalance", False):
            cautions.append("Consider models with class_weight support (e.g., RandomForest, SVC with weights).")
        if self.assumption_results.get("scaling_issues", False):
            cautions.append("Standardize features before using SVC, KNN, or LogisticRegression.")
        if self.assumption_results.get("separability", False):
            cautions.append("Linear models may struggle; consider tree-based or kernel methods.")
        return cautions
    
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

        assumptions = self.report.get("assumptions", self.assumption_results)

        for key, result in assumptions.items():
            print(f"\n🧠 {key.replace('_', ' ').title()}")
            print("-" * 30)

            if isinstance(result, dict):
                for subkey, value in result.items():
                    # Skip redundant missingness messages from individual checks
                    if subkey in ["notes", "recommendation"] and "missing" in str(value).lower():
                        continue
                    print(f"{subkey}: {value}")
            else:
                print(result)

        if "recommendations" in self.report:
            print("\n⚠️ Model Recommendations")
            print("-" * 30)
            for rec in self.report["recommendations"]:
                print(f"- {rec}")
