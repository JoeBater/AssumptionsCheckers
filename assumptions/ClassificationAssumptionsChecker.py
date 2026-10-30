import pandas as pd
import numpy as np
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier


from overlays.SharedOverlay import SharedOverlay
from overlays.ClassificationOverlay import ClassificationOverlay
from assumptions._common import NO_ALGORITHM_NOTE, verdict, print_check_result

LINEAR_MODELS = ["LogisticRegression"]
SCALE_SENSITIVE_MODELS = ["SVC", "KNeighborsClassifier"]
TREE_MODELS = ["DecisionTreeClassifier", "RandomForestClassifier", "GradientBoostingClassifier"]

CLASS_WEIGHT_ADVICE = "Use class_weight='balanced' to compensate for class imbalance."
IMBALANCE_ADVICE = {
    "LogisticRegression": CLASS_WEIGHT_ADVICE,
    "SVC": CLASS_WEIGHT_ADVICE,
    "DecisionTreeClassifier": CLASS_WEIGHT_ADVICE,
    "RandomForestClassifier": CLASS_WEIGHT_ADVICE,
    "GradientBoostingClassifier": "Pass sample_weight=compute_sample_weight('balanced', y) to fit() to compensate for class imbalance.",
    "KNeighborsClassifier": "KNN has no class weighting; consider resampling (e.g. SMOTE or undersampling) for class imbalance.",
}

class ClassificationAssumptionsChecker:
    def __init__(self, df, target, algorithm=None, visualize=False):
        # Overlays normalise missing-value tokens in place, so never touch the caller's frame
        self.df = df.copy()
        self.target = target
        self.algorithm = algorithm

        # Suppress individual overlay missingness warnings by setting to False temporarily
        # We'll generate a consolidated missingness report instead.
        # ClassificationOverlay normalises missing tokens, so build it before SharedOverlay.
        self.classification_overlay = ClassificationOverlay(self.df, target, visualize, suppress_missing_warnings=True)
        self.overlay = SharedOverlay(self.df, target, visualize, suppress_missing_warnings=True)
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
        """Run relevant checks depending on algorithm."""
        results = self.assumption_results = {}

        if self.algorithm is None:
            print(NO_ALGORITHM_NOTE)
            results["multicollinearity"] = self.overlay.check_multicollinearity()
            results["class_imbalance"] = self.overlay.check_class_imbalance()
            results["scaling_issues"] = self.overlay.check_scaling()
            results["redundancy"] = self.classification_overlay.check_redundancy()
            results["separability"] = self.classification_overlay.check_separability()
            self.recommend_models()
            return results

        if self.algorithm in LINEAR_MODELS:
            results["multicollinearity"] = self.overlay.check_multicollinearity()
            results["class_imbalance"] = self.overlay.check_class_imbalance()
            results["scaling_issues"] = self.overlay.check_scaling()
            results["redundancy"] = self.classification_overlay.check_redundancy()
            results["separability"] = self.classification_overlay.check_separability()

            has_multicollinearity = verdict(results["multicollinearity"], "has_multicollinearity")
            if has_multicollinearity:
                results["status"] = "conditionally_permissible"
                results["reason"] = (
                    "Multicollinearity detected: keep regularization on (the default L2 penalty) "
                    "and interpret coefficients with care, or drop/combine correlated features."
                )
            elif has_multicollinearity is None:
                results["status"] = "unknown"
                results["reason"] = "Multicollinearity check was skipped, so suitability could not be fully assessed."
            else:
                results["status"] = "suitable"
                results["reason"] = "No multicollinearity detected."
            if verdict(results["separability"], "separable") is False:
                results["reason"] += " Classes overlap in a 2D projection; a linear decision boundary may underfit."

            results["scaling_issues"]["required_for_model"] = True
            results["scaling_issues"]["notes"] += " Scaling is required for this algorithm: its penalty is scale-sensitive."

        elif self.algorithm in SCALE_SENSITIVE_MODELS:
            results["class_imbalance"] = self.overlay.check_class_imbalance()
            results["scaling_issues"] = self.overlay.check_scaling()
            if self.algorithm == "SVC":
                results["separability"] = self.classification_overlay.check_separability()

            results["status"] = "conditionally_permissible"
            results["reason"] = f"{self.algorithm} is suitable once features are scaled."
            results["scaling_issues"]["required_for_model"] = True
            results["scaling_issues"]["notes"] += " Scaling is required for this algorithm."

        elif self.algorithm in TREE_MODELS:
            results["tree_suitability"] = self.overlay.check_tree_suitability()
            results["class_imbalance"] = self.overlay.check_class_imbalance()
            results["scaling_issues"] = self.overlay.check_scaling()

            if not results["tree_suitability"]["tree_suitability"]:
                results["status"] = "conditionally_unsuitable"
                results["reason"] = results["tree_suitability"]["notes"]
            else:
                results["status"] = "suitable"
                results["reason"] = "Tree-based models do not assume linearity, scaling or uncorrelated features."
            results["scaling_issues"]["required_for_model"] = False
            results["scaling_issues"]["notes"] += " Scaling is optional for this algorithm."

        else:
            results["status"] = "unknown"
            results["reason"] = f"Algorithm {self.algorithm} not recognized."
            self.help_algorithms()

        if verdict(results.get("class_imbalance"), "is_imbalanced"):
            results["reason"] += " " + IMBALANCE_ADVICE[self.algorithm]

        self.report["assumptions"] = results
        return results

    def help_algorithms(self):
        print('ClassificationAssumptionsChecker')
        print('Checks multicollinearity, class imbalance, scaling, redundancy and separability for:')
        print(', '.join(LINEAR_MODELS))
        print('Checks scaling and class imbalance (plus separability for SVC) for:')
        print(', '.join(SCALE_SENSITIVE_MODELS))
        print('Checks tree-suitability and class imbalance for:')
        print(', '.join(TREE_MODELS))
        print("Use 'None' to run all checks and make recommendations")

    def recommend_models(self):
        """
        Recommend classification algorithms ordered by suitability based on assumption checks.
        Returns a list of dicts with algorithm, suitability, reason, and notes.
        """
        results = self.assumption_results
        
        # Extract check verdicts: True/False, or None when a check was skipped
        has_multicollinearity = verdict(results.get("multicollinearity"), "has_multicollinearity")
        is_imbalanced = verdict(results.get("class_imbalance"), "is_imbalanced")
        scaling_needed = verdict(results.get("scaling_issues"), "scaling_needed")
        is_separable = verdict(results.get("separability"), "separable")

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
            gb_notes.append("Use sample_weight (e.g. compute_sample_weight('balanced', y)) for imbalance")
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
        if is_separable is False:
            svc_notes.append("Kernel choice critical for overlapping classes")
        if is_separable is None:
            svc_reason = "Separability could not be assessed; effective with proper scaling and kernel choice."
        elif is_separable:
            svc_reason = "Effective for well-separated classes with proper scaling."
        else:
            svc_reason = "Can handle overlap with appropriate kernel."
        recommendations.append({
            "algorithm": "SVC",
            "suitability": svc_suitability,
            "reason": svc_reason,
            "notes": svc_notes
        })
        
        # KNeighborsClassifier - needs scaling
        knn_suitability = "high" if scaling_needed is False else "medium"
        knn_notes = ["Requires feature scaling"]
        if is_imbalanced:
            knn_notes.append("No class weighting available; consider resampling for imbalance")
        knn_reason = "Effective distance-based approach." if scaling_needed is False else "Distance-based method; sensitive to feature scaling."
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
        elif has_multicollinearity is None:
            lr_notes.append("Multicollinearity check skipped")
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
