
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
    def __init__(self, df, target=None, coverage_threshold=0.5, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.coverage_threshold = coverage_threshold

        # Replace common string missing tokens with np.nan
        self.X = df.drop(columns=[target]).copy()
        self.X.replace(to_replace=["na", "NA", "NaN", "missing", "?"], value=np.nan, inplace=True)
        self.y = df[target]

        # Missingness diagnostics
        self.X_complete = self.X.dropna()
        self.y_complete = self.y.loc[self.X_complete.index]
        self.coverage = len(self.X_complete) / len(self.X)
        self.has_missing = self.X.isna().any().any()
        self.missingness_note = (
            f"Only {self.coverage:.1%} of data is complete — some diagnostics may be skipped or caveated."
            if self.has_missing else "No missing values detected."
        )


    def check_separability(self, method="pca", perplexity=30, random_state=1223):
        """
        Projects features into 2D and checks class separability visually and via overlap metrics.
        """
        if self.coverage < self.coverage_threshold:
            return {
                "method": method,
                "separable": False,
                "overlap_score": None,
                "notes": f"Only {self.coverage:.1%} of data is complete — separability check skipped.",
                "recommendation": "Consider imputation or model-aware handling before assessing separability."
            }

        X = self.X_complete.values
        y = self.y_complete.values

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
            f"{self.missingness_note} Classes appear well-separated in 2D projection."
            if separable else f"{self.missingness_note} Significant class overlap — linear models may struggle."
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
            "recommendation": recommendation,
            "projection": df_proj.to_dict(orient="records")
        }




    def check_redundancy(self, threshold=0.95):
        """
        Checks for redundant features via correlation and mutual information clustering.
        """
        if self.coverage < self.coverage_threshold:
            return {
                "redundant_pairs": None,
                "mutual_info_scores": None,
                "notes": f"Only {self.coverage:.1%} of data is complete — redundancy check skipped.",
                "recommendation": "Consider imputation or column pruning before assessing redundancy."
            }

        X = self.X_complete.copy()
        y = self.y_complete.copy()

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
            f"{self.missingness_note} {len(redundant_pairs)} highly correlated feature pairs detected."
            if redundant_pairs else f"{self.missingness_note} No severe feature redundancy detected."
        ).strip()

        recommendation = (
            "Consider dropping or combining highly correlated features."
            if redundant_pairs else "No action needed for feature redundancy."
        )

        return {
            "redundant_pairs": redundant_pairs,
            "mutual_info_scores": mi_scores,
            "notes": notes,
            "recommendation": recommendation
        }
