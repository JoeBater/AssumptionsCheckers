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

from .ScalingOverlay import ScalingOverlay

class SharedOverlay:
    def __init__(self, df, target, visualize=False, missingness_threshold=0.5):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.missingness_threshold = missingness_threshold  # proportion
        self.pc_missing = self.df.isnull().mean().mean()  # proportion
        self.report = {}

        # Check for missing data
        self.df_no_missing = self.df.dropna()
        if self.df.isnull().any().any():
            self.pc_missing = (len(self.df) - len(self.df_no_missing)) / len(self.df) 
            self.report["missing_data"] =  (
                f"Warning: Data contains missing values ({self.pc_missing* 100:.2f}%). "
                + "\nTests will run on unmissing subset. Results may not generalize. "
                + "\nConsider imputation or removing rows/columns with excessive missingness."
            )
            print(self.report["missing_data"])

    # =======================================
    # Individual checks
    # =======================================
    

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
                        vif = float("inf")
                    except ValueError:
                        vif = float("inf")  # Use inf instead of None for consistency
                    vifs.append(vif)
                
                # Explicitly create DataFrame with proper types
                df = pd.DataFrame({
                    "feature": data.columns.tolist(),
                    "VIF": vifs
                })
                # Ensure VIF column is numeric
                df["VIF"] = pd.to_numeric(df["VIF"], errors='coerce')
                return df

        vif_data = compute_vif(X)
        too_high = vif_data[vif_data["VIF"] > threshold]

        drop_set = []
        X_temp = X.copy()

        while not too_high.empty and X_temp.shape[1] > 1:
            # choose feature with highest VIF to drop
            # Use explicit indexing to avoid type checker issues
            max_vif_idx = too_high["VIF"].idxmax()
            to_drop = too_high.loc[max_vif_idx, "feature"]
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
        Comprehensive scaling check using both ranges and standard deviations.
        Includes scaling method recommendations when scaling is needed.

        Parameters:
        - threshold: float, ratio above which scaling inconsistency is flagged

        Returns:
        - dict with scaling flag, ratios, feature details, recommendations, and notes
        """
        X = self.df.drop(columns=[self.target])
        X_numeric = X.select_dtypes(include=[np.number])
        
        if X_numeric.empty:
            return {
                "scaling_needed": False,
                "notes": "No numeric features found — scaling not applicable."
            }

        # Handle missing values
        if X_numeric.isnull().any().any():
            return {
                "scaling_needed": None,
                "notes": "Missing values detected — scaling check skipped.",
                "recommendation": "Impute or remove missing values before checking scaling."
            }

        # Handle infinite values
        if np.isinf(X_numeric.values).any():
            return {
                "scaling_needed": None,
                "notes": "Infinite values detected — scaling check skipped.",
                "recommendation": "Replace or remove infinite values before checking scaling."
            }
        
        # Compute statistics
        feature_ranges = X_numeric.max() - X_numeric.min()
        feature_stds = X_numeric.std()
        
        # Filter out zero variance features
        nonzero_stds = feature_stds[feature_stds > 0]
        nonzero_ranges = feature_ranges[feature_ranges > 0]

        if nonzero_stds.empty:
            return {
                "scaling_needed": False,
                "notes": "All features have zero variance — scaling not applicable."
            }

        # Calculate ratios
        std_ratio = nonzero_stds.max() / nonzero_stds.min()
        range_ratio = nonzero_ranges.max() / nonzero_ranges.min() if not nonzero_ranges.empty else 1.0
        
        # Determine if scaling is needed
        scaling_needed = std_ratio > threshold or range_ratio > threshold
        
        # Get feature names for detailed reporting
        max_std_feature = nonzero_stds.idxmax()
        min_std_feature = nonzero_stds.idxmin()
        
        # Build notes
        notes = (
            f"Feature '{max_std_feature}' has a standard deviation {std_ratio:.1f}× larger than '{min_std_feature}'. "
        )
        
        if scaling_needed:
            # Get scaling method recommendation from ScalingOverlay
            scaling_overlay = ScalingOverlay(X, threshold=int(threshold))
            scaler_recommendation = scaling_overlay.recommend_scaler()
            
            notes += (
                "Scaling is recommended to align feature influence. "
                f"Recommended scaler: {scaler_recommendation['recommended_scaler']}. "
                f"{scaler_recommendation['rationale']}"
            )
        else:
            notes += "Feature scales are reasonably aligned — scaling optional."

        # Visualization if requested
        if self.visualize:
            fig, ax = plt.subplots(1, 2, figsize=(12, 5))
            
            ax[0].bar(range(len(feature_ranges)), feature_ranges.values, color='teal')
            ax[0].set_title("Feature Ranges")
            ax[0].set_xlabel("Feature Index")
            ax[0].set_ylabel("Range")
            ax[0].set_xticks(range(len(feature_ranges)))
            ax[0].set_xticklabels(feature_ranges.index, rotation=45, ha='right')

            ax[1].bar(range(len(feature_stds)), feature_stds.values, color='orange')
            ax[1].set_title("Feature Standard Deviations")
            ax[1].set_xlabel("Feature Index")
            ax[1].set_ylabel("Std Dev")
            ax[1].set_xticks(range(len(feature_stds)))
            ax[1].set_xticklabels(feature_stds.index, rotation=45, ha='right')

            plt.tight_layout()
            plt.show()

        return {
            "scaling_needed": scaling_needed,
            "std_ratio": round(std_ratio, 2),
            "range_ratio": round(range_ratio, 2),
            "max_std_feature": max_std_feature,
            "min_std_feature": min_std_feature,
            "threshold": threshold,
            "notes": notes
        }


    




