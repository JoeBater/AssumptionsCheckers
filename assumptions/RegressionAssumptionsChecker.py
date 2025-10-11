import pandas as pd
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
        self.overlay = SharedOverlay(df, target, visualize, missingness_threshold=missingness_threshold)
        self.regression_overlay = RegressionOverlay(df, target, visualize, missingness_threshold=missingness_threshold)
        self.report = {}

        

    # =======================================
    # Orchestrator
    # =======================================
    def check_assumptions(self):
        """Run relevant checks depending on algorithm."""
        self.results = {}

        if self.algorithm in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
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


    def help_algorithms(self):
        print('RegressionAssumptionsChecker')
        print('Checks data assumptions (linearity, multicolinearity, heteroscedasticity) for the algorithms:')
        print('LinearRegression, Ridge, Lasso, ElasticNet')
        print('SVR, KNeighborsRegressor')
        print('DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor')

    def export_report(self, format="dict"):
        if format == "json":
            import json
            return json.dumps(self.report, indent=2)
        return self.report
    
    def report_assumptions(self):
        print("\n🔍 Assumption Diagnostics Report")
        print("=" * 40)

        assumptions = self.report.get("assumptions", self.results)

        for key, result in assumptions.items():
            print(f"\n🧠 {key.replace('_', ' ').title()}")
            print("-" * 30)

            if isinstance(result, dict):
                for subkey, value in result.items():
                    print(f"{subkey}: {value}")
            else:
                print(result)

        if "recommendations" in self.report:
            print("\n⚠️ Model Recommendations")
            print("-" * 30)
            for rec in self.report["recommendations"]:
                print(f"- {rec}")