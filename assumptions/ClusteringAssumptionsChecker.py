from overlays.ClusteringOverlay import ClusteringOverlay
from overlays.SharedOverlay import SharedOverlay
from assumptions._common import NO_ALGORITHM_NOTE, BaseAssumptionsChecker, verdict

ALGORITHMS = [
    "KMeans", "MiniBatchKMeans", "AgglomerativeClustering", "DBSCAN", "OPTICS",
    "GaussianMixture", "SpectralClustering",
]
ALIASES = {
    "kmeans": "KMeans", "k-means": "KMeans",
    "minibatchkmeans": "MiniBatchKMeans",
    "agglomerative": "AgglomerativeClustering", "agglomerativeclustering": "AgglomerativeClustering",
    "hierarchical": "AgglomerativeClustering",
    "dbscan": "DBSCAN", "optics": "OPTICS",
    "gmm": "GaussianMixture", "gaussianmixture": "GaussianMixture",
    "spectral": "SpectralClustering", "spectralclustering": "SpectralClustering",
}
NEEDS_K = {"KMeans", "MiniBatchKMeans", "AgglomerativeClustering", "GaussianMixture", "SpectralClustering"}

# Effect of a dataset characteristic on an algorithm, from worst to best:
# FAIL makes it unsuitable, CAP limits it to "low", each WARN lowers it one level from "high".
FAIL, CAP, WARN, NOTE, GOOD = "fail", "cap", "warn", "note", "good"

BASELINE = {
    "KMeans": "Fast, simple partitioning into compact clusters of similar size.",
    "MiniBatchKMeans": "K-Means on mini-batches: approximate, but scales to very large datasets.",
    "AgglomerativeClustering": "Builds a cluster hierarchy; the linkage choice sets which shapes it finds.",
    "DBSCAN": "Density-based: finds arbitrarily shaped clusters and labels noise.",
    "OPTICS": "Density-based like DBSCAN, but copes with clusters of different densities.",
    "GaussianMixture": "Models clusters as ellipsoids of any orientation and size, with soft membership.",
    "SpectralClustering": "Graph-based: recovers connected but non-convex clusters.",
}

# Notes that describe the algorithm rather than the data
PROPERTIES = {
    "KMeans": ["Needs n_clusters{k_hint}"],
    "MiniBatchKMeans": ["Needs n_clusters{k_hint}"],
    "AgglomerativeClustering": ["Needs n_clusters{k_hint}, or a distance_threshold to cut the dendrogram"],
    "DBSCAN": ["No n_clusters needed; tune eps from a k-distance plot with min_samples ≈ {min_samples}"],
    "OPTICS": ["No n_clusters or eps needed; set min_samples ≈ {min_samples}"],
    "GaussianMixture": ["Needs n_components{k_hint}", "Gives probabilistic (soft) membership via predict_proba"],
    "SpectralClustering": ["Needs n_clusters{k_hint}"],
}

KMEANS_FAMILY = ("KMeans", "MiniBatchKMeans")

# (name, condition on the facts, {algorithm: (effect, note)}). Notes may use {fact} placeholders.
RULES = [
    ("non_convex", lambda f: f["shape"] == "non_convex", {
        **{a: (FAIL, "Clusters are non-convex or not blob-shaped; K-Means only draws convex, roughly spherical clusters and cuts across them") for a in KMEANS_FAMILY},
        "GaussianMixture": (FAIL, "Clusters are non-convex or not blob-shaped; each Gaussian is an ellipsoid, so a mixture doesn't reproduce them"),
        "AgglomerativeClustering": (WARN, "Clusters are non-convex: use linkage='single' (ward, average and complete favour compact clusters), though single linkage chains through noise"),
        "DBSCAN": (GOOD, "Recovers the non-convex clusters found in the data"),
        "OPTICS": (GOOD, "Recovers the non-convex clusters found in the data"),
        "SpectralClustering": (GOOD, "A nearest-neighbour affinity recovers the non-convex clusters found in the data"),
    }),
    ("ellipsoidal", lambda f: f["shape"] == "ellipsoidal", {
        **{a: (FAIL, "Clusters are elongated or unequally sized; K-Means assumes spherical clusters of similar size and misassigns points") for a in KMEANS_FAMILY},
        "GaussianMixture": (GOOD, "Full-covariance components fit the elongated clusters found in the data"),
        "AgglomerativeClustering": (WARN, "Clusters are elongated; ward linkage favours spherical clusters, so try average linkage"),
    }),
    ("elongated", lambda f: f["elongated"] and f["shape"] != "ellipsoidal", {
        **{a: (WARN, "Clusters are elongated (longest axis {elongation}× the next); K-Means assumes spherical clusters and cuts across elongated ones unless they're well separated") for a in KMEANS_FAMILY},
        "AgglomerativeClustering": (WARN, "Clusters are elongated ({elongation}×); ward linkage favours spherical clusters, so try average linkage"),
        "GaussianMixture": (GOOD, "Full-covariance components fit elongated clusters"),
    }),
    ("globular", lambda f: f["shape"] == "globular", {
        **{a: (GOOD, "K-Means reproduces the separated dense regions in the data") for a in KMEANS_FAMILY},
        "AgglomerativeClustering": (GOOD, "Clusters are compact and globular, which suits ward linkage"),
        "GaussianMixture": (GOOD, "Clusters are compact; covariance_type='spherical' or 'diag' may suffice"),
    }),
    ("geometry_skipped", lambda f: not f["n_samples_ok"], {
        a: (WARN, "Cluster geometry could not be assessed: {skip_reason}") for a in ALGORITHMS
    }),
    ("shape_unknown", lambda f: f["shape"] is None and f["n_samples_ok"], {
        a: (NOTE, "Cluster shape could not be assessed (no separated dense regions), so convexity is unverified")
        for a in (*KMEANS_FAMILY, "GaussianMixture")
    }),
    ("overlapping", lambda f: f["overlapping"], {
        "DBSCAN": (WARN, "Clusters overlap or touch, so density-based methods tend to merge them into one"),
        "OPTICS": (WARN, "Clusters overlap or touch, so density-based methods tend to merge them into one"),
        "GaussianMixture": (GOOD, "Soft membership handles overlapping clusters"),
    }),
    ("unequal_variance", lambda f: f["unequal_variance"], {
        **{a: (WARN, "Cluster spreads differ by {spread_ratio}×; K-Means puts boundaries midway between centroids, so wide clusters lose points") for a in KMEANS_FAMILY},
        "AgglomerativeClustering": (WARN, "Cluster spreads differ by {spread_ratio}×; ward linkage favours clusters of similar size"),
        "GaussianMixture": (GOOD, "Each component has its own covariance, so unequal spreads are expected"),
    }),
    ("varying_density", lambda f: f["varying_density"], {
        "DBSCAN": (WARN, "Cluster densities differ ({spacing_ratio}× neighbour spacing); a single eps merges dense clusters or discards sparse ones as noise. OPTICS or HDBSCAN cope better"),
        "OPTICS": (GOOD, "Handles clusters of different densities"),
        "SpectralClustering": (WARN, "Cluster densities differ ({spacing_ratio}× neighbour spacing); a single RBF bandwidth fits them poorly, so try affinity='nearest_neighbors'"),
        "AgglomerativeClustering": (WARN, "Cluster densities differ ({spacing_ratio}× neighbour spacing); single linkage chains through the sparse clusters"),
        "GaussianMixture": (GOOD, "Components of different size and weight model density differences"),
    }),
    ("noise", lambda f: f["has_noise"], {
        **{a: (WARN, "{outlier_pct} of points are outliers; every point gets a cluster, and outliers drag centroids. Remove them first") for a in KMEANS_FAMILY},
        "GaussianMixture": (WARN, "{outlier_pct} of points are outliers, which inflate component covariances. Remove them first"),
        "AgglomerativeClustering": (WARN, "{outlier_pct} of points are outliers, which form singleton clusters or chain clusters together"),
        "SpectralClustering": (WARN, "{outlier_pct} of points are outliers, which distort the affinity graph"),
        "DBSCAN": (GOOD, "Labels the {outlier_pct} outliers as noise (-1)"),
        "OPTICS": (GOOD, "Labels the {outlier_pct} outliers as noise (-1)"),
    }),
    ("no_tendency", lambda f: f["tendency"] == "none", {
        a: (CAP, "Data shows no cluster tendency (Hopkins {hopkins}); any clusters found are likely arbitrary") for a in ALGORITHMS
    }),
    ("concentrated", lambda f: f["distances_concentrated"], {
        **{a: (WARN, "Distances are concentrated across {n_features} features; reduce dimensionality (PCA) first") for a in ALGORITHMS},
        "GaussianMixture": (WARN, "{n_features} features make full covariances unstable; use covariance_type='diag' or reduce dimensionality (PCA)"),
    }),
    ("large_n", lambda f: f["large_n"] and not f["very_large_n"], {
        "AgglomerativeClustering": (WARN, "{n_samples} rows: O(n²) memory and time; pass a connectivity graph or subsample"),
        "SpectralClustering": (WARN, "{n_samples} rows: the affinity matrix and eigendecomposition are expensive"),
        "OPTICS": (WARN, "{n_samples} rows: OPTICS is slow at this size; consider HDBSCAN"),
        "MiniBatchKMeans": (GOOD, "Scales well to {n_samples} rows"),
    }),
    ("very_large_n", lambda f: f["very_large_n"], {
        "AgglomerativeClustering": (FAIL, "{n_samples} rows: the O(n²) distance matrix will not fit in memory"),
        "SpectralClustering": (FAIL, "{n_samples} rows: the affinity matrix will not fit in memory"),
        "OPTICS": (WARN, "{n_samples} rows: OPTICS will be very slow; consider HDBSCAN"),
        "DBSCAN": (WARN, "{n_samples} rows: memory grows with neighbourhood size, so keep eps small"),
        "GaussianMixture": (WARN, "{n_samples} rows: EM is slow; fit on a sample"),
        "KMeans": (NOTE, "MiniBatchKMeans is much faster at {n_samples} rows"),
        "MiniBatchKMeans": (GOOD, "Scales well to {n_samples} rows"),
    }),
    ("moderate_n", lambda f: f["n_samples_ok"] and not f["large_n"], {
        "MiniBatchKMeans": (WARN, "Only {n_samples} rows: standard KMeans is exact and fast enough"),
    }),
    ("mixed", lambda f: f["mixed_types"], {
        **{a: (WARN, "Non-numeric columns present: needs numeric input. Consider K-Prototypes for mixed data") for a in KMEANS_FAMILY},
        "GaussianMixture": (WARN, "Non-numeric columns present: Gaussian components assume continuous features"),
        "SpectralClustering": (WARN, "Non-numeric columns present: build a precomputed affinity (e.g. from Gower distances)"),
        **{a: (WARN, "Non-numeric columns present: use metric='precomputed' with Gower distances")
           for a in ("AgglomerativeClustering", "DBSCAN", "OPTICS")},
    }),
    ("duplicates", lambda f: f["has_duplicates"], {
        "DBSCAN": (NOTE, "Duplicate rows inflate density and can create spurious core points"),
        "OPTICS": (NOTE, "Duplicate rows inflate density and can create spurious core points"),
        "AgglomerativeClustering": (NOTE, "Duplicate rows create zero-distance ties in the hierarchy"),
    }),
    ("tied", lambda f: f["has_tied_values"], {
        "DBSCAN": (NOTE, "Repeated values in {tied_list} stack points into dense slices that can become spurious clusters"),
        "OPTICS": (NOTE, "Repeated values in {tied_list} stack points into dense slices that can become spurious clusters"),
        "AgglomerativeClustering": (NOTE, "Repeated values in {tied_list} create many tied distances, so merge order is partly arbitrary"),
    }),
    ("scaling", lambda f: f["scaling_needed"], {
        a: (NOTE, "Standardise features first: their scales differ by {std_ratio}× and distances would be dominated by the largest")
        for a in ALGORITHMS
    }),
    ("missing", lambda f: f["has_missing"], {
        a: (NOTE, "Does not accept missing values; impute or drop incomplete rows") for a in ALGORITHMS
    }),
]

SUITABILITY_LEVELS = ["high", "medium", "low"]
STATUS = {
    "high": "suitable",
    "medium": "conditionally_permissible",
    "low": "conditionally_unsuitable",
    "unsuitable": "unsuitable",
}


class ClusteringAssumptionsChecker(BaseAssumptionsChecker):
    """Check whether a dataset's characteristics suit a clustering algorithm.

    There is no target: every column is treated as a feature. Pass ``algorithm`` to assess one
    algorithm, or leave it out to run every check and rank all supported algorithms.
    """

    def __init__(self, df, algorithm=None, max_samples=2000, random_state=0):
        super().__init__(df, target=None, algorithm=self._canonical(algorithm))
        self.overlay = ClusteringOverlay(self.df, suppress_missing_warnings=True,
                                         max_samples=max_samples, random_state=random_state)
        # Like the geometric checks, judge feature scales on complete rows rather than skipping
        self.shared_overlay = SharedOverlay(self.df.dropna(), None, suppress_missing_warnings=True)
        self.missingness_analysis = self._generate_missingness_analysis()

    @staticmethod
    def _canonical(algorithm):
        if algorithm is None:
            return None
        return ALIASES.get(str(algorithm).replace(" ", "").lower(), algorithm)

    def _run_checks(self):
        results = self.results = {}
        results["feature_types"] = self.overlay.check_feature_types()
        results["sample_size"] = self.overlay.check_sample_size()
        results["scaling_issues"] = self.shared_overlay.check_scaling()
        results["duplicates"] = self.overlay.check_duplicates()
        results["tied_values"] = self.overlay.check_tied_values()
        results["outliers"] = self.overlay.check_outliers()
        results["dimensionality"] = self.overlay.check_dimensionality()
        results["clustering_tendency"] = self.overlay.check_clustering_tendency()
        results["cluster_shape"] = self.overlay.check_cluster_shape()
        results["cluster_variance"] = self.overlay.check_cluster_variance()
        results["density_variation"] = self.overlay.check_density_variation()
        return results

    def _facts(self):
        """Flatten check results into the values the rules use; skipped checks give None."""
        r = self.results
        outliers = r["outliers"]
        n = r["sample_size"]["n_samples"]
        return {
            "shape": r["cluster_shape"].get("shape"),
            "overlapping": bool(r["cluster_shape"].get("overlapping")),
            "unequal_variance": verdict(r["cluster_variance"], "unequal_variance"),
            "spread_ratio": r["cluster_variance"].get("spread_ratio"),
            "elongated": verdict(r["cluster_variance"], "elongated"),
            "elongation": r["cluster_variance"].get("elongation"),
            # Separated dense regions count clusters of any shape; the mixture's k is Gaussian pieces
            "estimated_k": r["cluster_shape"].get("n_dense_regions") if r["cluster_shape"].get("shape")
                           else r["cluster_variance"].get("estimated_k"),
            "varying_density": verdict(r["density_variation"], "varying_density"),
            "spacing_ratio": r["density_variation"].get("spacing_ratio"),
            "has_noise": verdict(outliers, "has_noise"),
            "outlier_pct": f"{outliers['outlier_frac']:.1%}" if outliers.get("outlier_frac") is not None else None,
            "tendency": r["clustering_tendency"].get("strength"),
            "hopkins": r["clustering_tendency"].get("hopkins"),
            "distances_concentrated": verdict(r["dimensionality"], "distances_concentrated"),
            "n_features": r["sample_size"]["n_features"],
            "n_samples": f"{n:,}",
            "n_samples_ok": verdict(outliers, "has_noise") is not None,  # geometric checks ran
            "skip_reason": outliers["notes"].rstrip("."),
            "large_n": r["sample_size"]["large_n"],
            "very_large_n": r["sample_size"]["very_large_n"],
            "mixed_types": r["feature_types"]["mixed_types"],
            "has_duplicates": r["duplicates"]["has_duplicates"],
            "has_tied_values": verdict(r["tied_values"], "has_tied_values"),
            "tied_list": ", ".join(map(str, r["tied_values"]["tied_columns"])),
            "scaling_needed": verdict(r["scaling_issues"], "scaling_needed"),
            "std_ratio": r["scaling_issues"].get("std_ratio"),
            "has_missing": self.missingness_analysis["has_missing"],
        }

    def _assess(self, algorithm, facts):
        """Apply every rule to one algorithm and return its recommendation entry."""
        effects = []
        for _, condition, by_algorithm in RULES:
            if algorithm in by_algorithm and condition(facts):
                effect, note = by_algorithm[algorithm]
                effects.append((effect, note.format(**facts)))

        failures = [note for effect, note in effects if effect == FAIL]
        caps = [note for effect, note in effects if effect == CAP]
        warnings = [note for effect, note in effects if effect == WARN]
        if failures:
            suitability, reason = "unsuitable", failures[0] + "."
        elif caps:
            suitability, reason = "low", caps[0] + "."
        else:
            suitability = SUITABILITY_LEVELS[min(len(warnings), len(SUITABILITY_LEVELS) - 1)]
            reason = warnings[0] + "." if warnings else BASELINE[algorithm]

        k = facts["estimated_k"]
        hints = {
            "k_hint": f" (the data suggests about {k})" if k and k >= 2 else "",
            "min_samples": max(4, 2 * (facts["n_features"] or 2)),
        }
        order = {FAIL: 0, CAP: 1, WARN: 2, GOOD: 3, NOTE: 4}
        notes = [note for _, note in sorted(effects, key=lambda e: order[e[0]])]
        notes += [p.format(**hints) for p in PROPERTIES[algorithm]]
        return {"algorithm": algorithm, "suitability": suitability, "reason": reason, "notes": notes}

    def _caveats(self, facts):
        """Dataset-wide warnings that apply whichever algorithm is chosen."""
        caveats = []
        r = self.results
        if not facts["n_samples_ok"]:
            caveats.append(f"Cluster geometry could not be assessed: {facts['skip_reason']}.")
        if facts["tendency"] in ("none", "weak"):
            caveats.append(r["clustering_tendency"]["notes"])
        if facts["shape"] is None and r["cluster_shape"].get("notes"):
            caveats.append(r["cluster_shape"]["notes"])
        if facts["has_tied_values"]:
            if facts["shape"] in ("non_convex", "ellipsoidal"):
                caveats.append(
                    f"The '{facts['shape']}' shape verdict may come from repeated values in "
                    f"{facts['tied_list']} rather than real groups: check whether the dense regions "
                    "are slices of those columns before ruling out K-Means.")
            else:
                caveats.append(r["tied_values"]["notes"])
        if facts["distances_concentrated"]:
            caveats.append(r["dimensionality"]["notes"])
        if facts["mixed_types"]:
            caveats.append(r["feature_types"]["notes"])
        if facts["scaling_needed"]:
            caveats.append("Standardise features before clustering: every algorithm here is distance- or variance-based.")
        caveats.append("Validate the result: no single score proves clusters are real, so compare "
                       "silhouette scores, stability across seeds or subsamples, and domain sense.")
        return caveats

    def check_assumptions(self):
        """Run every diagnostic and assess the chosen algorithm, or rank them all if none was given."""
        results = self._run_checks()
        facts = self._facts()
        self.report["caveats"] = self._caveats(facts)

        if self.algorithm is None:
            print(NO_ALGORITHM_NOTE)
            self.recommend_models()
            self.report["assumptions"] = results
            return results

        if self.algorithm not in ALGORITHMS:
            results["status"] = "unknown"
            results["reason"] = f"Algorithm {self.algorithm} not recognized."
            self.help_algorithms()
        else:
            assessment = self._assess(self.algorithm, facts)
            results["status"] = STATUS[assessment["suitability"]]
            results["reason"] = assessment["reason"]
            results["advice"] = assessment["notes"]
        self.report["assumptions"] = results
        return results

    def recommend_models(self):
        """Rank every supported clustering algorithm by suitability for this dataset.

        Runs the checks first if check_assumptions() hasn't been called.
        """
        if "cluster_shape" not in self.results:
            self._run_checks()
            self.report["caveats"] = self._caveats(self._facts())
        facts = self._facts()
        return self._rank_recommendations([self._assess(a, facts) for a in ALGORITHMS])

    def help_algorithms(self):
        print("ClusteringAssumptionsChecker")
        print("Checks feature types, sample size, scaling, duplicates, tied values, outliers, dimensionality,")
        print("clustering tendency, cluster shape, cluster variance and density variation for:")
        print(", ".join(ALGORITHMS))
        print("Aliases such as 'GMM', 'Hierarchical' and 'Spectral' are accepted.")
        print("Use 'None' to run all checks and make recommendations")

    def report_assumptions(self):
        super().report_assumptions()
        if self.report.get("caveats"):
            print("\n📌 Caveats")
            print("-" * 30)
            for caveat in self.report["caveats"]:
                print(f"• {caveat}")
