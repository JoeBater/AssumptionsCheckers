"""Diagnostics describing whether a dataset suits particular clustering algorithms.

There is no target, so the geometric checks estimate cluster structure from the data itself:

- Density-connected components of a mutual k-nearest-neighbour graph (after dropping the
  sparsest 10% of points) act as a shape-agnostic reference partition.
- K-Means and a Gaussian mixture are fitted to the same points. If K-Means reproduces the
  components, clusters are globular; if only the mixture does, they are ellipsoidal (elongated
  or unequally sized, which the mixture must confirm); if neither does, they are non-convex.
- A Gaussian mixture chosen by BIC estimates the number of clusters and their relative spread.

All geometric checks run on standardised numeric features, after removing local outliers, on a
random sample of at most ``max_samples`` complete rows.
"""
from functools import cached_property

import numpy as np
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors, kneighbors_graph
from sklearn.preprocessing import StandardScaler

from ._common import MissingnessAwareOverlay, split_numeric

MIN_ROWS = 30                 # below this the geometric checks are skipped
NEIGHBOURS = 10               # k for the k-NN graph and neighbour spacing
LOF_NEIGHBOURS = 20
LOF_OUTLIER = 2.0             # local outlier factor above which a point counts as an outlier
NOISE_FRAC = 0.05             # outlier share treated as a noisy dataset
SOME_OUTLIERS_FRAC = 0.01
CORE_QUANTILE = 0.9           # keep the densest 90% of points when finding components
AGREEMENT = 0.8               # adjusted Rand index counted as "reproduces the components"
SPREAD_RATIO = 2.0            # cluster standard-deviation ratio flagged as unequal variance
ELONGATION = 4.0              # longest-to-next-longest cluster axis ratio flagged as elongated
SPACING_RATIO = 2.0           # neighbour-spacing ratio flagged as varying density
DISTANCE_CV = 0.15            # pairwise-distance coefficient of variation flagged as concentrated
HIGH_DIMENSIONAL = 20
DUPLICATE_FRAC = 0.01
TIED_SHARE = 0.2              # a column whose most common value covers this share is 'tied'
TIED_COLUMNS_FRAC = 0.1       # flag when at least this share of numeric columns is tied
LARGE_N = 10_000
VERY_LARGE_N = 50_000
HOPKINS_STRONG = 0.75
HOPKINS_WEAK = 0.6
HOPKINS_REPEATS = 5
FULL_COVARIANCE_MAX_DIM = 10  # beyond this BIC compares diagonal covariances only


class ClusteringOverlay(MissingnessAwareOverlay):
    def __init__(self, df, missingness_threshold=0.5, suppress_missing_warnings=False,
                 max_samples=2000, max_k=10, random_state=0, missing_tokens=None):
        super().__init__(
            df, None,
            missingness_threshold=missingness_threshold,
            suppress_missing_warnings=suppress_missing_warnings,
            missing_tokens=missing_tokens,
        )
        self.max_samples = max_samples
        self.max_k = max_k
        self.random_state = random_state

    # =======================================
    # Shared preparation (computed once)
    # =======================================

    @cached_property
    def _numeric(self):
        """Complete-case numeric features with non-finite values and constant columns removed."""
        X, _ = split_numeric(self.working_df)
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        return X.loc[:, X.std() > 0]

    @cached_property
    def _scaled(self):
        """Standardised numeric features, randomly subsampled to at most max_samples rows."""
        X = self._numeric.to_numpy(dtype=float)
        if len(X) > self.max_samples:
            rng = np.random.default_rng(self.random_state)
            X = X[rng.choice(len(X), self.max_samples, replace=False)]
        return StandardScaler().fit_transform(X) if len(X) else X

    @cached_property
    def _lof_scores(self):
        X = self._scaled
        lof = LocalOutlierFactor(n_neighbors=min(LOF_NEIGHBOURS, len(X) - 1))
        lof.fit(X)
        return -lof.negative_outlier_factor_

    @cached_property
    def _inliers(self):
        return self._scaled[self._lof_scores <= LOF_OUTLIER]

    @cached_property
    def _spacing(self):
        """Distance from each inlier to its k-th nearest neighbour (small = dense)."""
        X = self._inliers
        k = min(NEIGHBOURS, len(X) - 1)
        return NearestNeighbors(n_neighbors=k + 1).fit(X).kneighbors(X)[0][:, -1]

    @cached_property
    def _components(self):
        """Density-connected components: (core mask, labels of core points, sizeable-component mask)."""
        X = self._inliers
        graph = kneighbors_graph(X, min(NEIGHBOURS, len(X) - 1), include_self=False)
        graph = graph.minimum(graph.T).tocsr()  # mutual neighbours only, so thin bridges don't merge clusters
        # Drop the sparsest points (and their edges) rather than re-linking the rest: re-linking
        # would let the remaining points reach across the gaps between clusters
        core = self._spacing <= np.quantile(self._spacing, CORE_QUANTILE)
        _, labels = connected_components(graph[core][:, core], directed=False)
        sizes = np.bincount(labels)
        sizeable = sizes >= max(10, 0.05 * core.sum())
        return core, labels, sizeable

    @cached_property
    def _gmm(self):
        """Gaussian mixture with the number of components and covariance type chosen by BIC."""
        X = self._inliers
        covariance_types = ["full", "diag"] if X.shape[1] <= FULL_COVARIANCE_MAX_DIM else ["diag"]
        best, best_bic = None, np.inf
        for covariance_type in covariance_types:
            for k in range(1, min(self.max_k, len(X) // 5) + 1):
                gmm = GaussianMixture(k, covariance_type=covariance_type, n_init=2,
                                      random_state=self.random_state).fit(X)
                bic = gmm.bic(X)
                if bic < best_bic:
                    best, best_bic = gmm, bic
        return best

    def _skip(self, verdict_key, check_name):
        """Return a skipped-check result if the data can't support geometric checks, else None."""
        if self.too_much_missing(check_name):
            return {verdict_key: None, "notes": self.report["missing_data"]}
        if self._numeric.shape[1] == 0:
            return {verdict_key: None, "notes": f"No usable numeric features — {check_name} skipped."}
        if len(self._numeric) < MIN_ROWS:
            return {verdict_key: None,
                    "notes": f"Only {len(self._numeric)} complete rows (need {MIN_ROWS}) — {check_name} skipped."}
        return None

    # =======================================
    # Individual checks
    # =======================================

    def check_feature_types(self):
        """Clustering algorithms here need numeric features; flag categorical or mixed data."""
        _, excluded = split_numeric(self.df)
        numeric = [c for c in self.df.columns if c not in excluded]
        mixed = len(excluded) > 0
        if not numeric:
            notes = "No numeric features: these algorithms need numeric input."
        elif mixed:
            notes = (f"{len(excluded)} non-numeric column(s) {excluded} were excluded from the "
                     f"geometric checks, which use the {len(numeric)} numeric column(s).")
        else:
            notes = f"All {len(numeric)} features are numeric."
        return {
            "mixed_types": mixed,
            "numeric_features": numeric,
            "non_numeric_features": excluded,
            "notes": notes,
            "recommendation": (
                "Encode categoricals, or use a mixed-type method: Gower distances with "
                "metric='precomputed' (Agglomerative, DBSCAN, OPTICS) or K-Prototypes."
                if mixed else None
            ),
        }

    def check_sample_size(self):
        n, d = len(self.df), self._numeric.shape[1]
        large, very_large = n > LARGE_N, n > VERY_LARGE_N
        matrix_gb = n * n * 8 / 1e9
        if very_large:
            notes = (f"{n:,} rows: a full pairwise distance matrix would need ~{matrix_gb:,.0f} GB, "
                     "ruling out standard hierarchical and spectral clustering.")
        elif large:
            notes = (f"{n:,} rows: O(n²) methods (hierarchical, spectral, OPTICS) will be slow "
                     f"and memory-hungry (~{matrix_gb:,.1f} GB for a distance matrix).")
        else:
            notes = f"{n:,} rows and {d} numeric features: every algorithm is computationally feasible."
        return {
            "n_samples": n,
            "n_features": d,
            "large_n": large,
            "very_large_n": very_large,
            "notes": notes,
        }

    def check_duplicates(self):
        X = self.features
        frac = float(X.duplicated().mean()) if len(X) else 0.0
        flagged = frac > DUPLICATE_FRAC
        return {
            "has_duplicates": flagged,
            "duplicate_frac": round(frac, 4),
            "notes": (f"{frac:.1%} of rows are exact duplicates; they inflate local density."
                      if flagged else f"{frac:.1%} duplicate rows."),
            "recommendation": ("Deduplicate, or keep one copy of each row and pass the counts as "
                               "sample_weight (KMeans, DBSCAN)." if flagged else None),
        }

    def check_tied_values(self):
        """Flag columns dominated by one repeated value (counts, ordinal scales, one-hot flags).

        Points stacked on the same value form artificial dense slices, which density-based methods
        and the shape check can mistake for clusters.
        """
        X = self._numeric
        if X.empty:
            return {"has_tied_values": None, "tied_columns": {},
                    "notes": "No usable numeric features — tied-value check skipped."}
        shares = {col: float(X[col].value_counts(normalize=True).iloc[0]) for col in X.columns}
        tied = {col: round(share, 3) for col, share in shares.items() if share >= TIED_SHARE}
        flagged = len(tied) > 0 and len(tied) / X.shape[1] >= TIED_COLUMNS_FRAC
        described = ", ".join(f"{col} ({share:.0%})" for col, share in tied.items())
        if flagged:
            notes = (f"{len(tied)} of {X.shape[1]} numeric columns repeat one value heavily "
                     f"(most common value's share: {described}); stacked points form artificial "
                     "dense slices.")
        elif tied:
            notes = f"A few columns repeat one value heavily ({described}), too few to matter."
        else:
            notes = "No column is dominated by a single repeated value."
        return {
            "has_tied_values": flagged,
            "tied_columns": tied,
            "notes": notes,
            "recommendation": ("Treat low-cardinality columns as categorical, transform counts, or "
                               "check that clusters aren't just slices of these columns."
                               if flagged else None),
        }

    def check_outliers(self):
        skipped = self._skip("has_noise", "outlier check")
        if skipped:
            return skipped
        frac = float(np.mean(self._lof_scores > LOF_OUTLIER))
        has_noise = frac >= NOISE_FRAC
        if has_noise:
            notes = (f"{frac:.1%} of points are local outliers (LOF > {LOF_OUTLIER}): "
                     "the data contains background noise.")
        elif frac >= SOME_OUTLIERS_FRAC:
            notes = f"{frac:.1%} of points are local outliers (LOF > {LOF_OUTLIER})."
        else:
            notes = "No meaningful outliers detected."
        return {
            "has_noise": has_noise,
            "outlier_frac": round(frac, 4),
            "notes": notes,
            "recommendation": ("Use a method with a noise label (DBSCAN, OPTICS) or remove "
                               "outliers before partitioning." if has_noise else None),
        }

    def check_dimensionality(self):
        """High dimensionality makes distances concentrate, so 'nearest' stops meaning much."""
        skipped = self._skip("distances_concentrated", "dimensionality check")
        if skipped:
            return skipped
        X = self._scaled
        if len(X) > 500:
            X = X[np.random.default_rng(self.random_state).choice(len(X), 500, replace=False)]
        diffs = X[:, None, :] - X[None, :, :]
        distances = np.sqrt((diffs ** 2).sum(-1))[np.triu_indices(len(X), 1)]
        cv = float(distances.std() / distances.mean())
        d = X.shape[1]
        concentrated = cv < DISTANCE_CV
        if concentrated:
            notes = (f"Pairwise distances are concentrated (CV {cv:.2f} across {d} features): "
                     "points are nearly equidistant, so distance-based clusters are unreliable.")
        elif d >= HIGH_DIMENSIONAL:
            notes = f"{d} features, but distances still discriminate (CV {cv:.2f})."
        else:
            notes = f"{d} features; distances discriminate well (CV {cv:.2f})."
        return {
            "distances_concentrated": concentrated,
            "n_features": d,
            "distance_cv": round(cv, 3),
            "notes": notes,
            "recommendation": ("Reduce dimensionality (PCA, or UMAP for visual exploration) "
                               "before clustering." if concentrated or d >= HIGH_DIMENSIONAL else None),
        }

    @cached_property
    def _hopkins(self):
        """Hopkins statistic: ~0.5 for uniformly spread data, towards 1 for clustered data."""
        X = self._inliers
        n, d = X.shape
        m = max(10, min(n // 10, 200))
        nn = NearestNeighbors(n_neighbors=2).fit(X)
        rng = np.random.default_rng(self.random_state + 1)
        scores = []
        for _ in range(HOPKINS_REPEATS):
            data_nn = nn.kneighbors(X[rng.choice(n, m, replace=False)])[0][:, 1]
            uniform = rng.uniform(X.min(0), X.max(0), size=(m, d))
            uniform_nn = nn.kneighbors(uniform, n_neighbors=1)[0][:, 0]
            scores.append(uniform_nn.sum() / (uniform_nn.sum() + data_nn.sum()))
        return float(np.mean(scores))

    def check_clustering_tendency(self):
        """Hopkins statistic: ~0.5 for uniformly spread data, towards 1 for clustered data."""
        skipped = self._skip("has_cluster_tendency", "clustering tendency check")
        if skipped:
            return skipped
        hopkins = self._hopkins
        strength = "strong" if hopkins >= HOPKINS_STRONG else "weak" if hopkins >= HOPKINS_WEAK else "none"
        notes = {
            "strong": f"Hopkins statistic {hopkins:.2f}: the data has clear cluster structure.",
            "weak": (f"Hopkins statistic {hopkins:.2f}: only weak cluster structure; "
                     "clusters found may be arbitrary splits of one population."),
            "none": (f"Hopkins statistic {hopkins:.2f}: the data looks uniformly spread; "
                     "any algorithm will return clusters, but they won't mean much."),
        }[strength]
        return {
            "has_cluster_tendency": strength != "none",
            "hopkins": round(hopkins, 3),
            "strength": strength,
            "notes": notes,
        }

    def check_cluster_shape(self):
        """Classify cluster geometry as globular, ellipsoidal or non-convex.

        Compares K-Means and a full-covariance Gaussian mixture against density-connected components.
        """
        skipped = self._skip("non_globular", "cluster shape check")
        if skipped:
            return {**skipped, "shape": None}
        return dict(self._shape)

    @cached_property
    def _component_stats(self):
        """Spread ratio and elongation of the substantial components of the BIC-selected mixture.

        Both avoid the smallest eigenvalues: real data often has near-flat directions within a
        cluster (correlated features), which would swamp a volume or longest/shortest-axis ratio.
        """
        gmm = self._gmm
        if gmm.covariance_type == "diag":
            eigenvalues = gmm.covariances_
        else:
            eigenvalues = np.array([np.linalg.eigvalsh(c) for c in gmm.covariances_])
        eigenvalues = np.sort(np.maximum(eigenvalues[gmm.weights_ >= 0.05], 1e-12), axis=1)[:, ::-1]
        # Average variance per direction: how spread out each cluster is overall
        mean_variance = eigenvalues.mean(axis=1)
        spread = float(np.sqrt(mean_variance.max() / mean_variance.min())) if len(mean_variance) >= 2 else 1.0
        # Longest axis against the next longest: a cigar shape is what pulls K-Means off course.
        # Only assessed where BIC could consider full covariances; diagonal ones can't see rotation.
        if eigenvalues.shape[1] < 2 or self._inliers.shape[1] > FULL_COVARIANCE_MAX_DIM:
            elongation = None
        else:
            elongation = float(np.sqrt(eigenvalues[:, 0] / eigenvalues[:, 1]).max())
        return spread, elongation

    @cached_property
    def _shape(self):
        core, labels, sizeable = self._components
        n_components = int(sizeable.sum())
        estimated_k = self._gmm.n_components
        if n_components < 2:
            # Uniform data also fits several Gaussians, so only call it overlap if there's structure
            overlapping = estimated_k >= 2 and self._hopkins >= HOPKINS_WEAK
            return {
                "non_globular": None,
                "shape": None,
                "n_dense_regions": n_components,
                "overlapping": overlapping,
                "notes": (
                    f"No separated dense regions were found, though a Gaussian mixture suggests "
                    f"{estimated_k} clusters: clusters overlap or touch, so shape can't be assessed."
                    if overlapping else
                    "No separated dense regions were found, so cluster shape can't be assessed."
                ),
            }

        keep = sizeable[labels]
        X = self._inliers[core][keep]
        reference = labels[keep]
        kmeans_ari = adjusted_rand_score(
            reference, KMeans(n_components, n_init=10, random_state=self.random_state).fit_predict(X))
        # Mixtures can stall in a poor local optimum, so take the best of a few starts
        gmm_ari = max(
            adjusted_rand_score(reference, GaussianMixture(n_components, covariance_type="full",
                                                           init_params=init,
                                                           random_state=self.random_state + s)
                                .fit(X).predict(X))
            for init in ("kmeans", "k-means++") for s in range(3)
        )
        agreement = {"kmeans_agreement": round(kmeans_ari, 3), "gmm_agreement": round(gmm_ari, 3)}
        spread, elongation = self._component_stats
        elongated = elongation is not None and elongation > ELONGATION
        overlapping = False
        if kmeans_ari >= AGREEMENT:
            shape = "globular"
            notes = (f"{n_components} separated dense regions, which K-Means reproduces "
                     f"(ARI {kmeans_ari:.2f}): clusters are compact enough for K-Means.")
        elif gmm_ari >= AGREEMENT and (elongated or spread > SPREAD_RATIO):
            shape = "ellipsoidal"
            notes = (f"{n_components} separated dense regions that a Gaussian mixture reproduces "
                     f"(ARI {gmm_ari:.2f}) but K-Means does not (ARI {kmeans_ari:.2f}): clusters "
                     "are elongated or unequally sized.")
        elif gmm_ari >= AGREEMENT:
            # K-Means disagrees, yet the mixture's clusters are round and similar in size: most
            # likely a dense region holds several touching clusters, so K-Means was given too few
            shape, overlapping = None, True
            notes = (f"{n_components} separated dense regions that K-Means doesn't reproduce (ARI "
                     f"{kmeans_ari:.2f}), though clusters look round and similar in size: some "
                     "regions probably hold several touching clusters, so shape is inconclusive.")
        else:
            shape = "non_convex"
            notes = (f"{n_components} separated dense regions that neither K-Means (ARI "
                     f"{kmeans_ari:.2f}) nor a Gaussian mixture (ARI {gmm_ari:.2f}) reproduces: "
                     "clusters are non-convex or otherwise not blob-shaped (e.g. rings, "
                     "crescents, chains, or a tight clump inside a diffuse cloud).")
        return {
            "non_globular": None if shape is None else shape != "globular",
            "shape": shape,
            "n_dense_regions": n_components,
            "overlapping": overlapping,
            **agreement,
            "notes": notes,
        }

    def check_cluster_variance(self):
        """Compare the spread and elongation of clusters using the BIC-selected Gaussian mixture."""
        skipped = self._skip("unequal_variance", "cluster variance check")
        if skipped:
            return {**skipped, "estimated_k": None, "elongated": None}
        k = self._gmm.n_components
        if self._shape["shape"] == "non_convex":
            return {"estimated_k": k, "unequal_variance": None, "elongated": None,
                    "notes": ("Clusters aren't blob-shaped, so Gaussian components don't correspond "
                              "to clusters and their spreads aren't comparable.")}
        spread, elongation = self._component_stats
        unequal = spread > SPREAD_RATIO
        elongated = None if elongation is None else elongation > ELONGATION
        if k < 2:
            notes = "A single Gaussian describes the data best (BIC); there is no spread to compare."
        elif unequal:
            notes = f"A Gaussian mixture finds {k} clusters whose spreads differ by {spread:.1f}×."
        else:
            notes = f"A Gaussian mixture finds {k} clusters of similar spread (ratio {spread:.1f}×)."
        if elongated:
            notes += f" Clusters are elongated: the longest axis is {elongation:.1f}× the next."
        elif elongated is None:
            notes += (f" Elongation not assessed: with more than {FULL_COVARIANCE_MAX_DIM} features "
                      "only diagonal covariances are fitted, which can't detect rotated clusters.")
        return {
            "estimated_k": k,
            "unequal_variance": unequal,
            "spread_ratio": round(spread, 2),
            "elongated": elongated,
            "elongation": None if elongation is None else round(elongation, 2),
            "notes": notes,
        }

    def check_density_variation(self):
        """Compare typical neighbour spacing between clusters (larger spacing = sparser cluster)."""
        skipped = self._skip("varying_density", "density variation check")
        if skipped:
            return skipped
        core, labels, sizeable = self._components
        if sizeable.sum() >= 2:
            spacing = self._spacing[core]
            groups = [spacing[labels == c] for c in np.flatnonzero(sizeable)]
            source = "dense regions"
        else:
            gmm = self._gmm
            assignments = gmm.predict(self._inliers)
            groups = [self._spacing[assignments == c] for c in range(gmm.n_components)
                      if gmm.weights_[c] >= 0.05]
            source = "Gaussian mixture clusters"
        if len(groups) < 2:
            return {"varying_density": False, "spacing_ratio": 1.0,
                    "notes": "Only one cluster found — density variation not applicable."}
        medians = [np.median(g) for g in groups]
        ratio = float(max(medians) / min(medians))
        varying = ratio > SPACING_RATIO
        return {
            "varying_density": varying,
            "spacing_ratio": round(ratio, 2),
            "notes": (f"Typical neighbour spacing differs {ratio:.1f}× between {source}: "
                      "clusters have different densities." if varying else
                      f"Clusters have similar density (neighbour spacing ratio {ratio:.1f}× across {source})."),
        }
