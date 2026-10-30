from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, cdist

import matplotlib.pyplot as plt
import seaborn as sns

from ._common import DEFAULT_MISSING_TOKENS, MissingnessAwareOverlay, split_numeric


class ClassificationOverlay(MissingnessAwareOverlay):
    def __init__(self, df, target=None, visualize=False, custom_missing_values=None, missingness_threshold=0.5, suppress_missing_warnings=False):
        super().__init__(
            df, target, visualize,
            missingness_threshold=missingness_threshold,
            suppress_missing_warnings=suppress_missing_warnings,
            missing_tokens=custom_missing_values or DEFAULT_MISSING_TOKENS,
        )

    def check_separability(self, method="pca", perplexity=30, random_state=1223):
        """
        Projects features into 2D and checks class separability visually and via overlap metrics.
        Returns spatial relationships analysis instead of raw projection data.
        """
        if self.too_much_missing("Separability test"):
            return {
                "method": method,
                "separable": None,
                "overlap_score": None,
                "notes": f"Only {1.0 - self.row_missing_frac:.1%} of rows are complete — separability check skipped.",
                "data_used": None,
                "recommendation": "Consider imputation or model-aware handling before assessing separability."
            }

        df_to_use = self.working_df
        X, excluded = split_numeric(df_to_use.drop(columns=[self.target]))
        y = df_to_use[self.target]

        if X.shape[1] < 2:
            return {
                "method": method,
                "separable": None,
                "overlap_score": None,
                "notes": f"Fewer than two numeric features ({X.shape[1]}) — separability check skipped.",
                "data_used": self.data_used,
                "recommendation": None
            }

        # Standardise so the projection is not dominated by the largest-scale feature
        X_scaled = StandardScaler().fit_transform(X)
        reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state) if method == "tsne" else PCA(n_components=2)
        X_proj = reducer.fit_transform(X_scaled)

        df_proj = pd.DataFrame(X_proj, columns=["dim1", "dim2"])
        df_proj["label"] = y.to_numpy()  # positional: y keeps the gaps left by dropna()

        centroids = df_proj.groupby("label")[["dim1", "dim2"]].mean()
        spreads = df_proj.groupby("label")[["dim1", "dim2"]].std().mean(axis=1)  # Average spread across dimensions

        centroid_distances = pdist(centroids.values)
        mean_centroid_dist = np.mean(centroid_distances)
        mean_spread = spreads.mean()

        overlap_score = mean_spread / mean_centroid_dist if mean_centroid_dist > 0 else float("inf")
        separable = overlap_score < 0.5

        # Generate spatial relationship description
        relationship_desc = self._generate_spatial_description(method, centroids, spreads, mean_centroid_dist, mean_spread, separable)
        
        # Generate class summary table
        class_summary = self._generate_class_summary_table(centroids, spreads, centroid_distances)

        if self.visualize:
            plt.figure(figsize=(8, 6))
            sns.scatterplot(data=df_proj, x="dim1", y="dim2", hue="label", palette="Set2", alpha=0.7)
            plt.title(f"Class Separability ({method.upper()} projection)")
            plt.tight_layout()
            plt.show()

        recommendation = (
            "Linear models may be viable." if separable
            else "Consider tree-based or kernel methods due to class overlap."
        )

        result = {
            "method": method,
            "overlap_score": round(overlap_score, 3),
            "separable": separable,
            "spatial_description": relationship_desc,
            "class_summary": class_summary,
            "data_used": self.data_used,
            "recommendation": recommendation
        }
        if excluded:
            result["excluded_columns"] = excluded
        return result

    def _generate_spatial_description(self, method, centroids, spreads, mean_centroid_dist, mean_spread, separable):
        """Generate descriptive text about spatial relationships between class centroids."""
        method_name = method.upper()
        
        if separable:
            clustering_desc = "well-separated"
            overlap_desc = "minimal overlap"
            distance_desc = "high relative to within-class spread"
        else:
            clustering_desc = "tightly clustered"
            overlap_desc = "significant overlap"
            distance_desc = "low relative to within-class spread"
        
        return (
            f"In the {method_name} projection, class centroids are {clustering_desc} with {overlap_desc}. "
            f"The average inter-class distance is {distance_desc}, suggesting "
            f"{'good' if separable else 'poor'} separability."
        )

    def _generate_class_summary_table(self, centroids, spreads, centroid_distances):
        """Generate a summary table of class centroids, spreads, and overlap notes."""
        class_summary = []
        classes = centroids.index.tolist()
        
        # Calculate pairwise distances for overlap detection
        centroid_matrix = centroids.values
        pairwise_distances = cdist(centroid_matrix, centroid_matrix)
        
        for i, class_label in enumerate(classes):
            centroid = centroids.loc[class_label]
            spread = spreads.loc[class_label]
            
            # Find overlapping classes (those within 2 * spread distance)
            overlap_threshold = 2 * spread
            overlapping_classes = []
            
            for j, other_class in enumerate(classes):
                if i != j and pairwise_distances[i, j] < overlap_threshold:
                    overlapping_classes.append(str(other_class))
            
            # Generate notes
            if overlapping_classes:
                notes = f"Overlaps with {', '.join(overlapping_classes)}"
            else:
                if len(classes) > 1:
                    # Find nearest class
                    distances_to_others = [pairwise_distances[i, j] for j in range(len(classes)) if i != j]
                    min_distance = min(distances_to_others)
                    if min_distance > 3 * spread:
                        notes = "Well isolated"
                    else:
                        notes = "Slightly distinct"
                else:
                    notes = "Single class"
            
            class_summary.append({
                "Class": class_label,
                "Centroid (dim1, dim2)": f"({centroid.iloc[0]:.1f}, {centroid.iloc[1]:.1f})",
                "Spread": f"{spread:.1f}",
                "Notes": notes
            })
        
        return class_summary




    def check_redundancy(self, threshold=0.95):
        """
        Checks for redundant features via correlation and mutual information clustering.
        """
        if self.too_much_missing("Redundancy check"):
            return {
                "redundant_pairs": None,
                "mutual_info_scores": None,
                "notes": f"Only {1.0 - self.row_missing_frac:.1%} of rows are complete — redundancy check skipped.",
                "data_used": self.data_used,
                "recommendation": "Consider imputation or column pruning before assessing redundancy."
            }

        df_to_use = self.working_df
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
            "data_used": self.data_used,
            "recommendation": recommendation
        }
