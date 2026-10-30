from overlays.SharedOverlay import SharedOverlay
from overlays.RegressionOverlay import RegressionOverlay
from assumptions._common import NO_ALGORITHM_NOTE, BaseAssumptionsChecker, verdict

class RegressionAssumptionsChecker(BaseAssumptionsChecker):
    def __init__(self, df, target, algorithm=None, visualize=False, missingness_threshold=0.5):
        """
        Initialize the checker with data, target variable, and model parameters.
        """
        super().__init__(df, target, algorithm)
        # RegressionOverlay normalises missing tokens, so build it before SharedOverlay
        self.regression_overlay = RegressionOverlay(self.df, target, visualize, missingness_threshold=missingness_threshold, suppress_missing_warnings=True)
        self.overlay = SharedOverlay(self.df, target, visualize, missingness_threshold=missingness_threshold, suppress_missing_warnings=True)
        
        # Generate consolidated missingness analysis
        self.missingness_analysis = self._generate_missingness_analysis()

    # =======================================
    # Orchestrator
    # =======================================
    def check_assumptions(self):
        """Run relevant checks depending on algorithm."""
        self.results = {}

        if self.algorithm is None:
            print(NO_ALGORITHM_NOTE)
            self.recommend_models()

        elif self.algorithm in ["LinearRegression", "Ridge", "Lasso", "ElasticNet"]:
            self.results["linearity"] = self.regression_overlay.check_linearity()
            self.results["multicollinearity"] = self.regression_overlay.check_multicollinearity()
            self.results["homoscedasticity"] = self.regression_overlay.check_heteroscedasticity()
            self.results["scaling"] = self.overlay.check_scaling()
            self.results["normality"] = self.regression_overlay.check_residual_normality()
            #self.results["outliers"] = self.regression_overlay.check_outliers_influential_points()

            # Heteroscedasticity leaves coefficients unbiased but invalidates standard errors
            is_homoscedastic = verdict(self.results["homoscedasticity"], "homoscedasticity")
            if is_homoscedastic is False:
                self.results["status"] = "conditionally_permissible"
                self.results["reason"] = (
                    f"Residuals are heteroscedastic: {self.algorithm} coefficient estimates remain unbiased, "
                    "but standard errors and p-values are unreliable. Consider heteroscedasticity-robust "
                    "standard errors (e.g. HC3), transforming the target (e.g. log), or weighted least squares."
                )
            elif is_homoscedastic is None:
                self.results["status"] = "unknown"
                self.results["reason"] = (
                    "Homoscedasticity check was skipped (see the data quality summary), "
                    "so suitability could not be assessed."
                )
            else:
                self.results["status"] = "suitable"

            if self.algorithm == "LinearRegression":
                self.results["scaling"]["required_for_model"] = False
                self.results["scaling"]["notes"] += " Scaling is optional for ordinary least squares: predictions are unaffected by rescaling."
            else:
                self.results["scaling"]["required_for_model"] = True
                self.results["scaling"]["notes"] += " Scaling is required for this algorithm: its penalty is scale-sensitive."

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
        
        # Extract check verdicts: True/False, or None when a check was skipped
        is_linear = verdict(self.results.get("linearity"), "linearity")
        has_multicollinearity = verdict(self.results.get("multicollinearity"), "has_multicollinearity")
        is_homoscedastic = verdict(self.results.get("homoscedasticity"), "homoscedasticity")
        scaling_needed = verdict(self.results.get("scaling"), "scaling_needed")
        trees_suitable = verdict(self.results.get("tree_suitability"), "tree_suitability")

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
        
        # LinearRegression - best case scenario; skipped checks count as unknown, not violated
        lr_checks = {"Linearity": is_linear, "Homoscedasticity": is_homoscedastic,
                     "Multicollinearity": None if has_multicollinearity is None else not has_multicollinearity}
        lr_all_hold = all(ok is True for ok in lr_checks.values())
        lr_violated = any(ok is False for ok in lr_checks.values())
        lr_suitability = "high" if lr_all_hold else "low" if lr_violated else "medium"
        lr_notes = ["No scaling required (OLS predictions are unaffected by rescaling)"]
        if has_multicollinearity:
            lr_notes.append("High multicollinearity detected - model will be unstable")
        if is_homoscedastic is False:
            lr_notes.append("Heteroscedasticity detected")
        if is_linear is False:
            lr_notes.append("Nonlinearity detected")
        lr_notes.extend(f"{name} check skipped" for name, ok in lr_checks.items() if ok is None)
        if lr_all_hold:
            lr_reason = "Ideal when linearity and homoscedasticity assumptions hold."
        elif lr_violated:
            lr_reason = "Assumptions violated; consider regularized or nonlinear alternatives."
        else:
            lr_reason = "Some assumption checks were skipped; suitability could not be fully assessed."
        recommendations.append({
            "algorithm": "LinearRegression",
            "suitability": lr_suitability,
            "reason": lr_reason,
            "notes": lr_notes
        })
        
        # Ridge - handles multicollinearity
        ridge_suitability = "high" if has_multicollinearity else "medium"
        ridge_notes = ["Requires scaling", "Handles multicollinearity via L2 regularization"]
        ridge_reason = "Ridge regression controls multicollinearity through regularization." if has_multicollinearity else "Regularized linear model with good bias-variance tradeoff."
        recommendations.append({
            "algorithm": "Ridge",
            "suitability": ridge_suitability,
            "reason": ridge_reason,
            "notes": ridge_notes
        })
        
        # Lasso - feature selection
        lasso_suitability = "medium"
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
        svr_suitability = "high" if scaling_needed is False else "medium"
        svr_notes = ["Requires feature scaling (e.g. StandardScaler)", "Effective for nonlinear patterns"]
        if is_linear is False:
            svr_notes.append("RBF kernel handles nonlinearity well")
        svr_reason = "Strong choice for nonlinear regression." if scaling_needed is False else "Kernel methods handle nonlinearity effectively with proper scaling."
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
        
        return self._rank_recommendations(recommendations)

    def help_algorithms(self):
        print('RegressionAssumptionsChecker')
        print('Checks linearity, multicolinearity, heteroscedasticity assumptions for the algorithms:')
        print('LinearRegression, Ridge, Lasso, ElasticNet')
        print('Checks scaling for:')
        print('SVR, KNeighborsRegressor')
        print('Checks tree-suitability for:')
        print('DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor')
        print("Use 'None' to run all checks and make recommendations")
