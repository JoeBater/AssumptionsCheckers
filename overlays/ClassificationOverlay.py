
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

class ClassificationOverlay:
    def __init__(self, df, target=None, visualize=False, custom_missing_values=None, missingness_threshold=0.5):
        self.df = df
        self.target = target
        self.visualize = visualize # not used currently
        self.report = {}
        self.pc_missing = 0.0
        self.missingness_threshold = missingness_threshold

        # Replace common string missing tokens with np.nan
        default_missing_values = ["na", "NA", "NaN", "missing", "?", ".", "-999", "-9999"]
        missing_values = set(custom_missing_values) if custom_missing_values else set(default_missing_values)   
        self.df.replace(to_replace=missing_values, value=np.nan, inplace=True)
        self.df_no_missing = self.df.dropna()
        if self.df.isnull().any().any():
            self.pc_missing = (len(self.df) - len(self.df_no_missing)) / len(self.df) 
            self.report["missing_data"] =  (
                f"Warning: Data contains missing values ({self.pc_missing* 100:.2f}%). "
                + "\nTests will run on unmissing subset. Results may not generalize. "
                + "\nConsider imputation or removing rows/columns with excessive missingness."
            )
            print(self.report["missing_data"])


    def check_separability(self, method="pca", perplexity=30, random_state=1223):
        """
        Projects features into 2D and checks class separability visually and via overlap metrics.
        """
        if self.pc_missing > self.missingness_threshold:
            self.report["missing_data"] = (
                f"Error: Over {self.pc_missing:.2f}% of data is missing. "
                "\nSeparability test cannot proceed."
            )
            return {
                "method": method,
                "separable": False,
                "overlap_score": None,
                "notes": f"Only {self.pc_missing:.1%} of data is complete — separability check skipped.",
                "data_used": None,
                "recommendation": "Consider imputation or model-aware handling before assessing separability."
            }

        # Use unmissing data only if missingness exists
        df_to_use = self.df_no_missing if self.pc_missing > 0 else self.df
        X = df_to_use.drop(columns=[self.target])
        y = df_to_use[self.target]

        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state) if method == "tsne" else PCA(n_components=2)
        X_proj = reducer.fit_transform(X)

        df_proj = pd.DataFrame(X_proj, columns=["dim1", "dim2"])
        df_proj["label"] = y

        centroids = df_proj.groupby("label")[["dim1", "dim2"]].mean()
        spreads = df_proj.groupby("label")[["dim1", "dim2"]].std()

        centroid_distances = pdist(centroids.values)
        mean_centroid_dist = np.mean(centroid_distances)
        mean_spread = spreads.values.mean()

        overlap_score = mean_spread / mean_centroid_dist if mean_centroid_dist > 0 else float("inf")
        separable = overlap_score < 0.5

        if self.visualize:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_proj, x="dim1", y="dim2", hue="label", palette="Set2", alpha=0.7)
            plt.title(f"Class Separability ({method.upper()} projection)")
            plt.tight_layout()
            plt.show()

        notes = (
            f"Classes appear well-separated in 2D projection."
            if separable else f"Significant class overlap — linear models may struggle."
        ).strip()

        recommendation = (
            "Linear models may be viable." if separable
            else "Consider tree-based or kernel methods due to class overlap."
        )

        return {
            "method": method,
            "overlap_score": round(overlap_score, 3),
            "separable": separable,
            "notes": notes,
            "data_used": "Unmissing subset" if self.pc_missing > 0 else "Full dataset",
            "recommendation": recommendation,
            "projection": df_proj.to_dict(orient="records")
        }




    def check_redundancy(self, threshold=0.95):
        """
        Checks for redundant features via correlation and mutual information clustering.
        """
        if self.pc_missing > self.missingness_threshold:
            self.report["missing_data"] = (
                f"Error: Over {self.pc_missing:.2f}% of data is missing. "
                "\nSeparability test cannot proceed."
            )
            return {
                "redundant_pairs": None,
                "mutual_info_scores": None,
                "notes": f"Only {1.0 - self.pc_missing:.1%} of data is complete — redundancy check skipped.",
                "data_used": "Unmissing subset" if self.pc_missing > 0 else "Full dataset",
                "recommendation": "Consider imputation or column pruning before assessing redundancy."
            }

        # Use unmissing data only if missingness exists
        df_to_use = self.df_no_missing if self.pc_missing > 0 else self.df
        X = df_to_use.drop(columns=[self.target])
        y = df_to_use[self.target]

        for col in X.select_dtypes(include=["object", "category"]).columns:
            X[col] = LabelEncoder().fit_transform(X[col])

        corr_matrix = X.corr().abs()
        redundant_pairs = [
            (i, j, round(corr_matrix.loc[i, j], 3))
            for i in corr_matrix.columns
            for j in corr_matrix.columns
            if i != j and corr_matrix.loc[i, j] > threshold
        ]

        mi = mutual_info_classif(X, y, discrete_features="auto")

        mi_scores = dict(zip(X.columns, np.round(mi, 3)))

        if self.visualize:
            plt.figure(figsize=(8, 6))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Feature Correlation Matrix")
            plt.tight_layout()
            plt.show()

        notes = (
            f"{len(redundant_pairs)} highly correlated feature pairs detected."
            if redundant_pairs else f"No severe feature redundancy detected."
        ).strip()

        recommendation = (
            "Consider dropping or combining highly correlated features."
            if redundant_pairs else "No action needed for feature redundancy."
        )

        return {
            "redundant_pairs": redundant_pairs,
            "mutual_info_scores": mi_scores,
            "notes": notes,
            "data_used": "Unmissing subset" if self.pc_missing > 0 else "Full dataset",
            "recommendation": recommendation
        }
