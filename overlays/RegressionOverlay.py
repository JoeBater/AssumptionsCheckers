import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_breuschpagan

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

from statsmodels.stats.outliers_influence import variance_inflation_factor

import pandas as pd
import numpy as np
import warnings
from collections import Counter
from scipy.stats import shapiro
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import seaborn as sns

class RegressionOverlay:
    def __init__(self, df, target, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.X = df.drop(columns=[target])
        self.y = df[target]

    # =======================================
    # Individual checks
    # =======================================
    def check_linearity(self):
        
        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        reset_test = linear_reset(model, power=2, use_f=True)

        if self.visualize:
            sm.graphics.plot_partregress_grid(model)
            plt.suptitle("Partial Residuals")
            plt.show()

        return {
            "linearity": reset_test.pvalue >= 0.05,
            "p_value": reset_test.pvalue,
            "notes": "Linearity assumption holds." if reset_test.pvalue >= 0.05 else "Non-linearity detected."
        }

    def check_residual_normality(self):
        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        stat, p_value = shapiro(model.resid)

        if self.visualize:
            sm.qqplot(model.resid, line='s')
            plt.title("Q-Q Plot of Residuals")
            plt.show()

        return {
            "normality": p_value >= 0.05,
            "p_value": p_value,
            "notes": "Residuals appear normal." if p_value >= 0.05 else "Residuals deviate from normality."
        }

    def check_scaling_need(self, threshold=10):
        stds = self.X.std()
        ratio = stds.max() / stds.min()

        return {
            "scaling_needed": ratio > threshold,
            "std_ratio": ratio,
            "notes": "Scaling recommended." if ratio > threshold else "Feature scales are aligned."
        }

    def check_multicollinearity(self, threshold=10):
        """Check multicollinearity using VIF, only if 2+ predictors.
        Returns diagnostics and minimal drop set recommendation.
        """
        X = self.df.drop(columns=[self.target])

        if X.shape[1] < 2:
            return {
                "multicollinearity": True,
                "details": None,
                "notes": "Only one predictor — multicollinearity check not applicable.",
                "recommendation": None
            }



        def compute_vif(data):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vifs = []
                for i in range(data.shape[1]):
                    try:
                        vif = variance_inflation_factor(data.values, i)
                    except np.linalg.LinAlgError:
                        vif = float("inf")  # or np.nan if you prefer
                    except ValueError:
                        vif = None
                    vifs.append(vif)
                return pd.DataFrame({
                    "feature": data.columns,
                    "VIF": vifs
                })

        vif_data = compute_vif(X)
        too_high = vif_data[vif_data["VIF"] > threshold]

        drop_set = []
        X_temp = X.copy()

        while not too_high.empty and X_temp.shape[1] > 1:
            # choose feature with highest VIF to drop
            to_drop = too_high.sort_values("VIF", ascending=False)["feature"].iloc[0]
            drop_set.append(to_drop)
            X_temp = X_temp.drop(columns=[to_drop])

            if X_temp.shape[1] == 0:
                break  # Prevent zero-size array crash

            vif_data = compute_vif(X_temp)
            too_high = vif_data[vif_data["VIF"] > threshold]

        if X_temp.shape[1] == 0:
            return {
                "multicollinearity": False,
                "details": [],
                "notes": "All predictors dropped due to perfect multicollinearity.",
                "recommendation": "Rebuild feature set to avoid linear dependence.",
                "drop_set": drop_set
            }
        
        recommendation = (
            f"High or perfect multicollinearity detected. "
            f"Consider removing these features: {drop_set}" if drop_set else None
        )
        return {
            "multicollinearity": len(drop_set) == 0,
            "details": vif_data.to_dict(orient="records"),
            "notes": (
                "Perfect or high multicollinearity detected" if drop_set 
                else "No severe multicollinearity."
            ),
            "recommendation": recommendation,
            "drop_set": drop_set
        }




    def check_heteroscedasticity(self):
        """Check heteroscedasticity using Breusch-Pagan test."""


        X = self.df.drop(columns=[self.target])
        y = self.df[self.target]
        X_const = sm.add_constant(X)
        model = sm.OLS(y, X_const).fit()

        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)
        return {
            "homoscedasticity": lm_pvalue >= 0.05,
            "p_value": lm_pvalue,
            "notes": (
                "Homoscedasticity assumption holds." if lm_pvalue >= 0.05
                else "Residuals are heteroscedastic."
            )
        }

    def check_tree_suitability(self):
        X = self.df.drop(columns=[self.target])
        n_samples, n_features = X.shape

        issues = []

        if n_samples < 50:
            issues.append("Very small dataset — tree-based models may overfit.")
        if any(X[col].nunique() > n_samples * 0.5 for col in X.select_dtypes(include="object")):
            issues.append("High-cardinality categorical features may cause overfitting.")
        if n_features > n_samples:
            issues.append("More features than samples — trees may struggle.")

        return {
            "tree_suitability": len(issues) == 0,
            "notes": " ".join(issues) if issues else "No dataset characteristics preclude tree-based models."
        }

    def check_class_imbalance(self, threshold=0.2):
        """
        Checks for class imbalance in the target variable y.

        Parameters:
        - y: array-like target labels
        - threshold: float, ratio below which imbalance is flagged
        - plot: bool, whether to show a class distribution plot

        Returns:
        - dict with imbalance flag, class counts, and imbalance ratio
        """
        y = np.array(self.df[self.target])
        if y.ndim == 0:
            raise ValueError("Target must be array-like, not scalar.")
        y = np.ravel(y)

        class_counts = Counter(y)
        total = sum(class_counts.values())
        ratios = {cls: count / total for cls, count in class_counts.items()}
        
        # Find minority/majority ratio
        sorted_ratios = sorted(ratios.values())
        imbalance_ratio = sorted_ratios[0] / sorted_ratios[-1] if len(sorted_ratios) > 1 else 1.0
        is_imbalanced = imbalance_ratio < threshold

        if self.visualize:
            pd.Series(class_counts).sort_index().plot(kind='bar', color='skyblue')
            plt.title("Class Distribution")
            plt.xlabel("Class")
            plt.ylabel("Count")
            plt.tight_layout()
            plt.show()

        return {
            "is_imbalanced": is_imbalanced,
            "class_counts": dict(class_counts),
            "imbalance_ratio": round(imbalance_ratio, 3),
            "threshold": threshold
        }
    
    def check_scaling(self, threshold=10.0):
        """
        Checks for feature scaling issues by comparing ranges and standard deviations.

        Parameters:
        - threshold: float, ratio above which scaling inconsistency is flagged

        Returns:
        - dict with scaling flag, feature ranges, stds, and max ratio
        """
        X = np.array(self.X)

        # Compute range and std per feature
        feature_ranges = np.ptp(X, axis=0)  # peak-to-peak (max - min)
        feature_stds = np.std(X, axis=0)

        # Avoid divide-by-zero
        safe_ranges = np.where(feature_ranges == 0, 1e-8, feature_ranges)
        safe_stds = np.where(feature_stds == 0, 1e-8, feature_stds)

        range_ratio = np.max(safe_ranges) / np.min(safe_ranges)
        std_ratio = np.max(safe_stds) / np.min(safe_stds)
        is_unscaled = range_ratio > threshold or std_ratio > threshold

        if self.visualize:
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))
            ax[0].bar(range(len(feature_ranges)), feature_ranges, color='teal')
            ax[0].set_title("Feature Ranges")
            ax[0].set_xlabel("Feature Index")
            ax[0].set_ylabel("Range")

            ax[1].bar(range(len(feature_stds)), feature_stds, color='orange')
            ax[1].set_title("Feature Standard Deviations")
            ax[1].set_xlabel("Feature Index")
            ax[1].set_ylabel("Std Dev")

            plt.tight_layout()
            plt.show()

        return {
            "is_unscaled": is_unscaled,
            "range_ratio": round(range_ratio, 2),
            "std_ratio": round(std_ratio, 2),
            "threshold": threshold,
            "notes": (
                "Scaling recommended due to inconsistent feature magnitudes."
                if is_unscaled else "Feature scales appear aligned."
            )
        }


    

    def check_separability(self, method="pca", perplexity=30, random_state=42):
        """
        Projects features into 2D and checks class separability visually and via overlap metrics.

        Parameters:
        - method: "pca" or "tsne"
        - perplexity: t-SNE parameter (ignored for PCA)
        - random_state: reproducibility

        Returns:
        - dict with method used, overlap score, and notes
        """
        X = self.X.values if isinstance(self.X, pd.DataFrame) else self.X
        y = self.y.values if isinstance(self.y, pd.Series) else self.y

        if method == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
        else:
            reducer = PCA(n_components=2)

        X_proj = reducer.fit_transform(X)

        # Compute overlap score: mean distance between class centroids vs intra-class spread
        df_proj = pd.DataFrame(X_proj, columns=["dim1", "dim2"])
        df_proj["label"] = y

        centroids = df_proj.groupby("label")[["dim1", "dim2"]].mean()
        spreads = df_proj.groupby("label")[["dim1", "dim2"]].std()

        # Compute pairwise centroid distances
        
        centroid_distances = pdist(centroids.values)
        mean_centroid_dist = np.mean(centroid_distances)
        mean_spread = spreads.values.mean()

        overlap_score = mean_spread / mean_centroid_dist if mean_centroid_dist > 0 else float("inf")
        separable = overlap_score < 0.5  # Tunable threshold

        if self.visualize:
            
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_proj, x="dim1", y="dim2", hue="label", palette="Set2", alpha=0.7)
            plt.title(f"Class Separability ({method.upper()} projection)")
            plt.tight_layout()
            plt.show()

        return {
            "method": method,
            "overlap_score": round(overlap_score, 3),
            "separable": separable,
            "notes": (
                "Classes appear well-separated in 2D projection."
                if separable else "Significant class overlap — linear models may struggle."
            )
        }



    def check_redundancy(self, threshold=0.95, task="classification"):
        """
        Checks for redundant features via correlation and mutual information clustering.

        Parameters:
        - threshold: float, correlation above which features are flagged as redundant
        - task: "classification" or "regression"

        Returns:
        - dict with flagged pairs, mutual info scores, and notes
        """
        X = self.X.copy()

        # Encode categorical features if needed
        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        # Correlation matrix
        corr_matrix = X.corr().abs()
        redundant_pairs = [
            (i, j, corr_matrix.loc[i, j])
            for i in corr_matrix.columns
            for j in corr_matrix.columns
            if i != j and corr_matrix.loc[i, j] > threshold
        ]

        # Mutual information
        y = self.y
        if task == "classification":
            mi = mutual_info_classif(X, y, discrete_features="auto")
        else:
            mi = mutual_info_regression(X, y)

        mi_scores = dict(zip(X.columns, np.round(mi, 3)))

        if self.visualize:
            import seaborn as sns
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.show()

        return {
            "redundant_pairs": redundant_pairs,
            "mutual_info_scores": mi_scores,
            "notes": (
                f"{len(redundant_pairs)} highly correlated feature pairs detected."
                if redundant_pairs else "No severe feature redundancy detected."
            )
        }
