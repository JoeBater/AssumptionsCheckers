import numpy as np

class ScalingOverlay:
    def __init__(self, X, threshold=10):
        self.X = X.select_dtypes(include=[np.number])
        self.threshold = threshold
        self.report = {}

    def check_scaling_need(self):
        stds = self.X.std()
        nonzero_stds = stds[stds > 0]

        if nonzero_stds.empty:
            self.report["scaling_needed"] = False
            self.report["std_ratio"] = None
            self.report["notes"] = "All features have zero variance — scaling not applicable."
            return self.report

        ratio = nonzero_stds.max() / nonzero_stds.min()
        max_feature = nonzero_stds.idxmax()
        min_feature = nonzero_stds.idxmin()

        self.report.update({
            "scaling_needed": ratio > self.threshold,
            "std_ratio": round(ratio, 2),
            "max_feature": max_feature,
            "min_feature": min_feature,
            "notes": (
                f"Feature '{max_feature}' has a standard deviation {ratio:.1f}× larger than '{min_feature}'. "
                "Scaling is recommended to align feature influence."
                if ratio > self.threshold else
                "Feature scales are reasonably aligned — scaling optional."
            )
        })
        return self.report

    def recommend_scaler(self):
        skewness = self.X.skew()
        outlier_sensitive = skewness[skewness.abs() > 2].index.tolist()

        if outlier_sensitive:
            scaler = "RobustScaler"
            rationale = f"Features {outlier_sensitive} show high skew — RobustScaler is recommended."
        elif skewness.abs().mean() < 0.5:
            scaler = "StandardScaler"
            rationale = "Feature distributions are roughly normal — StandardScaler is appropriate."
        else:
            scaler = "PowerTransformer"
            rationale = "Moderate skew detected — PowerTransformer may improve symmetry."

        return {
            "recommended_scaler": scaler,
            "skew_summary": skewness.round(2).to_dict(),
            "outlier_sensitive_features": outlier_sensitive,
            "rationale": rationale
        }
