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

class SharedOverlay:
    def __init__(self, df, target, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.X = df.drop(columns=[target])
        self.y = df[target]

    # =======================================
    # Individual checks
    # =======================================
    
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

        if X.isna().any().any():
            return {
                "multicollinearity": True,
                "details": None,
                "notes": "Missing data — multicollinearity check skipped.",
                "recommendation": "Missing values detected, downstream classifiers like XGBoost or CatBoost may tolerate missingness natively. Consider whether imputation is necessary based on your modeling choice."
            }
        
        non_numeric = [col for col in X.columns if not np.issubdtype(X[col].dtype, np.number)]
        if non_numeric:
            return {
                "multicollinearity": True,
                "details": None,
                "notes": f"Non-numeric columns present — multicollinearity check skipped: {non_numeric}",
                "recommendation": "Recast or exclude non-numeric columns before running VIF."
            }
        

      
        # Checks for infinite values in the dataset.  Returns a summary of affected columns and counts.
        inf_mask = np.isinf(self.df.select_dtypes(include=[np.number]))
        inf_counts = inf_mask.sum()

        flagged = inf_counts[inf_counts > 0]
        if not flagged.empty:
            return {
                "has_infinite": True,
                "columns": flagged.index.tolist(),
                "total": int(flagged.sum()),
                "notes": f"Infinite values detected in {len(flagged)} column(s).",
                "recommendation": "Replace or remove infinite values before running numeric diagnostics."
            }


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
        X_df = self.X.select_dtypes(include=[np.number])
        X = X_df.values

        if np.isnan(X).any():
            return {
                "is_scaled": True,
                "details": None,
                "notes": "Missing values detected — scaling check skipped.",
                "recommendation": "Impute or remove missing values before checking scaling."
            }

        if np.isinf(X).any():
            return {
                "is_scaled": True,
                "details": None,
                "notes": "Infinite values detected — scaling check skipped.",
                "recommendation": "Replace or remove infinite values before checking scaling."
            }
        
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


    




