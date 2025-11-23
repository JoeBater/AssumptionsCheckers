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
        """
        Recommend classification algorithms ordered by suitability based on assumption checks.
        Returns a list of dicts with algorithm, suitability, reason, and notes.
        """
        results = self.assumption_results
        
        # Extract check results
        multicollinearity = results.get("multicollinearity", {})
        has_multicollinearity = multicollinearity.get("multicollinearity", False) if isinstance(multicollinearity, dict) else multicollinearity
        
        class_imbalance = results.get("class_imbalance", {})
        is_imbalanced = class_imbalance.get("is_imbalanced", False) if isinstance(class_imbalance, dict) else class_imbalance
        
        scaling = results.get("scaling_issues", {})
        scaling_needed = scaling.get("scaling_needed", False) if isinstance(scaling, dict) else scaling
        
        separability = results.get("separability", {})
        is_separable = separability.get("separable", True) if isinstance(separability, dict) else separability
        
        recommendations = []
        
        # RandomForestClassifier - generally robust
        rf_notes = ["No scaling required", "Handles multicollinearity well"]
        if is_imbalanced:
            rf_notes.append("Supports class_weight for imbalance")
        rf_suitability = "high"
        rf_reason = "Robust ensemble method handles most scenarios well."
        recommendations.append({
            "algorithm": "RandomForestClassifier",
            "suitability": rf_suitability,
            "reason": rf_reason,
            "notes": rf_notes
        })
        
        # GradientBoostingClassifier - also robust
        gb_notes = ["No scaling required", "Strong predictive power"]
        if is_imbalanced:
            gb_notes.append("Use scale_pos_weight for imbalance")
        recommendations.append({
            "algorithm": "GradientBoostingClassifier",
            "suitability": "high",
            "reason": "Powerful ensemble with good generalization.",
            "notes": gb_notes
        })
        
        # SVC - depends on scaling and separability
        svc_suitability = "high" if is_separable else "medium"
        svc_notes = ["Requires StandardScaler or MinMaxScaler"]
        if is_imbalanced:
            svc_notes.append("Use class_weight parameter for imbalance")
        if not is_separable:
            svc_notes.append("Kernel choice critical for overlapping classes")
        svc_reason = "Effective for well-separated classes with proper scaling." if is_separable else "Can handle overlap with appropriate kernel."
        recommendations.append({
            "algorithm": "SVC",
            "suitability": svc_suitability,
            "reason": svc_reason,
            "notes": svc_notes
        })
        
        # KNeighborsClassifier - needs scaling
        knn_suitability = "medium" if scaling_needed else "high"
        knn_notes = ["Requires feature scaling"]
        if is_imbalanced:
            knn_notes.append("Consider weighted neighbors for imbalance")
        knn_reason = "Distance-based method; sensitive to feature scaling." if scaling_needed else "Effective distance-based approach."
        recommendations.append({
            "algorithm": "KNeighborsClassifier",
            "suitability": knn_suitability,
            "reason": knn_reason,
            "notes": knn_notes
        })
        
        # LogisticRegression - problems with multicollinearity
        lr_suitability = "low" if has_multicollinearity else "medium"
        lr_notes = ["Requires scaling"]
        if has_multicollinearity:
            lr_notes.append("High multicollinearity detected")
            lr_notes.append("Use regularization: L2 (Ridge) or L1 (Lasso)")
            lr_notes.append("Consider dropping or combining correlated features")
        if is_imbalanced:
            lr_notes.append("Use class_weight parameter")
        lr_reason = "Linear model struggles with multicollinearity without regularization." if has_multicollinearity else "Baseline linear classifier."
        recommendations.append({
            "algorithm": "LogisticRegression",
            "suitability": lr_suitability,
            "reason": lr_reason,
            "notes": lr_notes
        })
        
        # DecisionTreeClassifier - simple but prone to overfitting
        dt_notes = ["No scaling required", "Interpretable results", "Prone to overfitting on complex data"]
        recommendations.append({
            "algorithm": "DecisionTreeClassifier",
            "suitability": "medium",
            "reason": "Simple and interpretable but prone to overfitting.",
            "notes": dt_notes
        })
        
        # Sort by suitability (high > medium > low > unsuitable)
        suitability_order = {"high": 3, "medium": 2, "low": 1, "unsuitable": 0}
        recommendations.sort(key=lambda x: suitability_order.get(x["suitability"], 0), reverse=True)
        
        self.report["recommendations"] = recommendations
        return recommendations
    
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
            print("\n⚠️ Model Recommendations (ordered by suitability)")
            print("-" * 30)
            for rec in self.report["recommendations"]:
                print(f"\n• {rec['algorithm']} [{rec['suitability'].upper()}]")
                print(f"  Reason: {rec['reason']}")
                if rec.get('notes'):
                    for note in rec['notes']:
                        print(f"  - {note}")
