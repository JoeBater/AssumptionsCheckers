import pandas as pd
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
        self.overlay = SharedOverlay(df, target, visualize)
        self.classification_overlay = ClassificationOverlay(df, target, visualize)
        self.report = {}
        

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

        assumptions = self.report.get("assumptions", self.assumption_results)

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
