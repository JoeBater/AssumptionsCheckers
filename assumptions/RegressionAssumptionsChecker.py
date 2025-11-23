import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from overlays.SharedOverlay import SharedOverlay
from overlays.RegressionOverlay import RegressionOverlay

class RegressionAssumptionsChecker:
    def __init__(self, df, target, algorithm=None, visualize=False, missingness_threshold=0.5):
        """
        Initialize the checker with data, target variable, and model parameters.
        """
        self.df = df
        self.target = target
        self.algorithm = algorithm
        self.overlay = SharedOverlay(df, target, visualize, missingness_threshold=missingness_threshold, suppress_missing_warnings=True)
        self.regression_overlay = RegressionOverlay(df, target, visualize, missingness_threshold=missingness_threshold)
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

        

    # =======================================
    # Orchestrator
    # =======================================
    def check_assumptions(self):
        """Run relevant checks depending on algorithm."""
        self.results = {}

        if self.algorithm is None:
            self.recommend_models()

        elif self.algorithm in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
            self.results["linearity"] = self.regression_overlay.check_linearity()
            self.results["multicollinearity"] = self.regression_overlay.check_multicollinearity()
            self.results["homoscedasticity"] = self.regression_overlay.check_heteroscedasticity()
            self.results["scaling"] = self.overlay.check_scaling()
            self.results["normality"] = self.regression_overlay.check_residual_normality()
            #self.results["outliers"] = self.regression_overlay.check_outliers_influential_points()

            # Evaluate hard constraints
            if not self.results["homoscedasticity"]["homoscedasticity"]:
                self.results["status"] = "unsuitable"
                self.results["reason"] = (
                    f"{self.algorithm} is not suitable because residuals are heteroscedastic. "
                    "Consider robust alternatives such as HuberRegressor or RANSACRegressor."
                )
            else:
                self.results["status"] = "suitable"

            self.results["scaling"]["required_for_model"] = True
            self.results["scaling"]["notes"] += " Scaling is required for this algorithm."

        elif self.algorithm in ["SVR", "KNeighborsRegressor"]:
            scaling_check = self.overlay.check_scaling()
            self.results["scaling"] = scaling_check

            self.results["status"] = "conditionally_permissible"
            self.results["scaling"]["required_for_model"] = True
            self.results["scaling"]["notes"] += " Scaling is required for this algorithm."


        elif self.algorithm in ["DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]:
            tree_check = self.overlay.check_tree_suitability()
            scaling_check = self.overlay.check_scaling()

            self.results["tree_suitability"] = tree_check
            self.results["scaling"] = scaling_check

            if not tree_check["tree_suitability"]:
                self.results["status"] = "conditionally_unsuitable"
                self.results["reason"] = tree_check["notes"]
            else:
                self.results["status"] = "suitable"
                self.results["reason"] = "Tree-based models do not assume linearity or homoscedasticity."

            self.results["scaling"]["required_for_model"] = False
            self.results["scaling"]["notes"] += " Scaling is optional for this algorithm."

        else:
            self.results["status"] = "unknown"
            self.results["reason"] = f"Algorithm {self.algorithm} not recognized."
            self.help_algorithms()

        self.report["assumptions"] = self.results
        return self.results

    def recommend_models(self):
        """
        Recommend regression algorithms ordered by suitability based on assumption checks.
        Returns a list of dicts with algorithm, suitability, reason, and notes.
        """
        self.results["linearity"] = self.regression_overlay.check_linearity()
        self.results["multicollinearity"] = self.regression_overlay.check_multicollinearity()
        self.results["homoscedasticity"] = self.regression_overlay.check_heteroscedasticity()
        self.results["scaling"] = self.overlay.check_scaling()
        self.results["normality"] = self.regression_overlay.check_residual_normality()
        self.results["tree_suitability"] = self.overlay.check_tree_suitability()
        
        # Extract check results
        linearity = self.results.get("linearity", {})
        is_linear = linearity.get("linearity", False) if isinstance(linearity, dict) else linearity
        
        multicollinearity = self.results.get("multicollinearity", {})
        has_multicollinearity = not multicollinearity.get("multicollinearity", True) if isinstance(multicollinearity, dict) else False
        
        homoscedasticity = self.results.get("homoscedasticity", {})
        is_homoscedastic = homoscedasticity.get("homoscedasticity", False) if isinstance(homoscedasticity, dict) else False
        
        scaling = self.results.get("scaling", {})
        scaling_needed = scaling.get("scaling_needed", False) if isinstance(scaling, dict) else False
        
        normality = self.results.get("normality", {})
        is_normal = normality.get("normality", False) if isinstance(normality, dict) else False
        
        tree_suitability = self.results.get("tree_suitability", {})
        trees_suitable = tree_suitability.get("tree_suitability", False) if isinstance(tree_suitability, dict) else False
        
        recommendations = []
        
        # RandomForestRegressor - robust ensemble
        rf_suitability = "high" if trees_suitable else "medium"
        rf_notes = ["No scaling required", "Handles nonlinearity well"]
        if has_multicollinearity:
            rf_notes.append("Robust to multicollinearity")
        rf_reason = "Robust ensemble method handles nonlinearity and complex patterns." if trees_suitable else "Good general-purpose method but data characteristics may limit performance."
        recommendations.append({
            "algorithm": "RandomForestRegressor",
            "suitability": rf_suitability,
            "reason": rf_reason,
            "notes": rf_notes
        })
        
        # GradientBoostingRegressor - powerful ensemble
        gb_suitability = "high" if trees_suitable else "medium"
        gb_notes = ["No scaling required", "Strong predictive power", "Requires hyperparameter tuning"]
        gb_reason = "Powerful ensemble with excellent generalization." if trees_suitable else "Effective but data characteristics may require careful tuning."
        recommendations.append({
            "algorithm": "GradientBoostingRegressor",
            "suitability": gb_suitability,
            "reason": gb_reason,
            "notes": gb_notes
        })
        
        # LinearRegression - best case scenario
        lr_suitability = "high" if (is_linear and is_homoscedastic and not has_multicollinearity) else "low"
        lr_notes = ["Requires scaling" if scaling_needed else "Scaling optional"]
        if has_multicollinearity:
            lr_notes.append("High multicollinearity detected - model will be unstable")
        if not is_homoscedastic:
            lr_notes.append("Heteroscedasticity detected")
        if not is_linear:
            lr_notes.append("Nonlinearity detected")
        lr_reason = "Ideal when linearity and homoscedasticity assumptions hold." if (is_linear and is_homoscedastic) else "Assumptions violated; consider regularized or nonlinear alternatives."
        recommendations.append({
            "algorithm": "LinearRegression",
            "suitability": lr_suitability,
            "reason": lr_reason,
            "notes": lr_notes
        })
        
        # Ridge - handles multicollinearity
        ridge_suitability = "high" if has_multicollinearity else "medium"
        ridge_notes = ["Requires scaling", "Handles multicollinearity via L2 regularization"]
        if not is_homoscedastic:
            ridge_notes.append("More robust to heteroscedasticity than OLS")
        ridge_reason = "Ridge regression controls multicollinearity through regularization." if has_multicollinearity else "Regularized linear model with good bias-variance tradeoff."
        recommendations.append({
            "algorithm": "Ridge",
            "suitability": ridge_suitability,
            "reason": ridge_reason,
            "notes": ridge_notes
        })
        
        # Lasso - feature selection
        lasso_suitability = "medium" if has_multicollinearity else "medium"
        lasso_notes = ["Requires scaling", "Performs feature selection via L1 regularization"]
        if has_multicollinearity:
            lasso_notes.append("Can select among correlated features")
        recommendations.append({
            "algorithm": "Lasso",
            "suitability": lasso_suitability,
            "reason": "L1 regularization enables automatic feature selection.",
            "notes": lasso_notes
        })
        
        # ElasticNet - combines Ridge and Lasso
        elasticnet_notes = ["Requires scaling", "Combines L1 and L2 regularization"]
        if has_multicollinearity:
            elasticnet_notes.append("Good balance for correlated features")
        recommendations.append({
            "algorithm": "ElasticNet",
            "suitability": "medium",
            "reason": "Combines L1 and L2 regularization for balanced feature handling.",
            "notes": elasticnet_notes
        })
        
        # SVR - needs scaling
        svr_suitability = "medium" if scaling_needed else "high"
        svr_notes = ["Requires StandardScaler or RBF scaler", "Effective for nonlinear patterns"]
        if not is_linear:
            svr_notes.append("RBF kernel handles nonlinearity well")
        svr_reason = "Kernel methods handle nonlinearity effectively with proper scaling." if scaling_needed else "Strong choice for nonlinear regression."
        recommendations.append({
            "algorithm": "SVR",
            "suitability": svr_suitability,
            "reason": svr_reason,
            "notes": svr_notes
        })
        
        # KNeighborsRegressor - needs scaling
        knn_suitability = "low" if scaling_needed else "medium"
        knn_notes = ["Requires feature scaling", "Distance-based method", "Sensitive to feature scale differences"]
        knn_reason = "Distance-based approach requires careful scaling." if scaling_needed else "Can be effective but requires proper scaling."
        recommendations.append({
            "algorithm": "KNeighborsRegressor",
            "suitability": knn_suitability,
            "reason": knn_reason,
            "notes": knn_notes
        })
        
        # DecisionTreeRegressor - simple baseline
        dt_notes = ["No scaling required", "Interpretable results", "Prone to overfitting"]
        recommendations.append({
            "algorithm": "DecisionTreeRegressor",
            "suitability": "medium",
            "reason": "Simple baseline; consider ensemble methods for better performance.",
            "notes": dt_notes
        })
        
        # Sort by suitability (high > medium > low > unsuitable)
        suitability_order = {"high": 3, "medium": 2, "low": 1, "unsuitable": 0}
        recommendations.sort(key=lambda x: suitability_order.get(x["suitability"], 0), reverse=True)
        
        self.report["recommendations"] = recommendations
        return recommendations

    def help_algorithms(self):
        print('RegressionAssumptionsChecker')
        print('Checks linearity, multicolinearity, heteroscedasticity assumptions for the algorithms:')
        print('LinearRegression, Ridge, Lasso, ElasticNet')
        print('Checks scaling for:')
        print('SVR, KNeighborsRegressor')
        print('Checks tree-suitability for:')
        print('DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor')
        print("Use 'None' to run all checks and make recommendations")

    def export_report(self, format="dict"):
        if format == "json":
            import json
            return json.dumps(self.report, indent=2)
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