from overlays.SharedOverlay import SharedOverlay
from overlays.ClassificationOverlay import ClassificationOverlay
from assumptions._common import NO_ALGORITHM_NOTE, BaseAssumptionsChecker, verdict

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

class ClassificationAssumptionsChecker(BaseAssumptionsChecker):
    def __init__(self, df, target, algorithm=None, visualize=False):
        super().__init__(df, target, algorithm)

        # Suppress individual overlay missingness warnings by setting to False temporarily
        # We'll generate a consolidated missingness report instead.
        # ClassificationOverlay normalises missing tokens, so build it before SharedOverlay.
        self.classification_overlay = ClassificationOverlay(self.df, target, visualize, suppress_missing_warnings=True)
        self.overlay = SharedOverlay(self.df, target, visualize, suppress_missing_warnings=True)
        
        # Generate consolidated missingness analysis
        self.missingness_analysis = self._generate_missingness_analysis()
        

    @property
    def assumption_results(self):
        """Backwards-compatible alias for self.results."""
        return self.results

    def check_assumptions(self):
        """Run relevant checks depending on algorithm."""
        results = self.results = {}

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
        results = self.results
        
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
        
        return self._rank_recommendations(recommendations)
