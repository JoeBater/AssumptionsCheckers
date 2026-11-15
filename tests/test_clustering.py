"""The clustering checker is tested on synthetic data whose geometry is known by construction."""
import json

import numpy as np
import pandas as pd
import pytest
from sklearn import datasets

from assumptions._common import verdict
from assumptions.ClusteringAssumptionsChecker import ClusteringAssumptionsChecker

LEVELS = {"high": 3, "medium": 2, "low": 1, "unsuitable": 0}


def _frame(X):
    return pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])


def _blobs():
    return _frame(datasets.make_blobs(500, centers=3, random_state=1)[0])


def _anisotropic():
    X = datasets.make_blobs(500, centers=3, random_state=170)[0]
    return _frame(X @ [[0.6, -0.6], [-0.4, 0.8]])


def _varied():
    return _frame(datasets.make_blobs(500, cluster_std=[1.0, 2.5, 0.5], random_state=170)[0])


def _moons():
    return _frame(datasets.make_moons(500, noise=0.05, random_state=0)[0])


def _circles():
    return _frame(datasets.make_circles(500, factor=0.5, noise=0.05, random_state=0)[0])


def _blobs_with_noise():
    rng = np.random.default_rng(0)
    X = datasets.make_blobs(450, centers=3, random_state=1)[0]
    noise = rng.uniform(X.min(0) - 3, X.max(0) + 3, size=(50, 2))
    return _frame(np.vstack([X, noise]))


def _run(df, **kwargs):
    checker = ClusteringAssumptionsChecker(df, **kwargs)
    results = checker.check_assumptions()
    return checker, results


def _suitability(checker):
    return {r["algorithm"]: r["suitability"] for r in checker.report["recommendations"]}


def test_kmeans_is_a_good_choice_for_globular_blobs():
    checker, results = _run(_blobs())
    assert results["cluster_shape"]["shape"] == "globular"
    assert results["cluster_shape"]["n_dense_regions"] == 3
    suitability = _suitability(checker)
    assert suitability["KMeans"] == "high"
    assert suitability["GaussianMixture"] == "high"


@pytest.mark.parametrize("make", [_moons, _circles], ids=["moons", "circles"])
def test_kmeans_is_unsuitable_for_non_convex_clusters(make):
    checker, results = _run(make())
    assert results["cluster_shape"]["shape"] == "non_convex"
    suitability = _suitability(checker)
    for algorithm in ("KMeans", "MiniBatchKMeans", "GaussianMixture"):
        assert suitability[algorithm] == "unsuitable"
    for algorithm in ("DBSCAN", "OPTICS", "SpectralClustering"):
        assert suitability[algorithm] == "high"


def test_elongated_clusters_suit_gmm_but_not_kmeans():
    checker, results = _run(_anisotropic())
    assert results["cluster_shape"]["shape"] == "ellipsoidal"
    suitability = _suitability(checker)
    assert suitability["KMeans"] == "unsuitable"
    assert suitability["GaussianMixture"] == "high"


def test_well_separated_elongated_clusters_still_warn_kmeans():
    # K-Means recovers these, but only because they're far apart, so it's a warning, not a failure
    X = datasets.make_blobs(500, centers=3, random_state=7)[0] @ [[0.6, -0.6], [-0.4, 0.8]]
    checker, results = _run(_frame(X))
    assert results["cluster_variance"]["elongated"] is True
    suitability = _suitability(checker)
    assert suitability["KMeans"] == "medium"
    assert suitability["GaussianMixture"] == "high"


def test_touching_round_clusters_are_inconclusive_not_ellipsoidal():
    # Two of these blobs touch and form one dense region; that alone must not condemn K-Means
    X = datasets.make_blobs(500, centers=3, random_state=2, cluster_std=0.8)[0]
    checker, results = _run(_frame(X))
    assert results["cluster_shape"]["shape"] is None
    assert results["cluster_shape"]["overlapping"] is True
    assert _suitability(checker)["KMeans"] != "unsuitable"


def test_unequal_variance_and_density_favour_gmm_and_optics():
    checker, results = _run(_varied())
    assert results["cluster_variance"]["unequal_variance"] is True
    assert results["density_variation"]["varying_density"] is True
    suitability = _suitability(checker)
    assert suitability["GaussianMixture"] == "high"
    assert suitability["OPTICS"] == "high"
    assert LEVELS[suitability["KMeans"]] < LEVELS["high"]
    assert LEVELS[suitability["DBSCAN"]] < LEVELS[suitability["OPTICS"]]


def test_background_noise_favours_density_based_methods():
    checker, results = _run(_blobs_with_noise())
    assert results["outliers"]["has_noise"] is True
    suitability = _suitability(checker)
    assert suitability["DBSCAN"] == "high"
    assert LEVELS[suitability["KMeans"]] < LEVELS[suitability["DBSCAN"]]


def test_uniform_data_has_no_cluster_tendency():
    df = _frame(np.random.default_rng(42).uniform(size=(500, 2)))
    checker, results = _run(df)
    assert results["clustering_tendency"]["strength"] == "none"
    assert results["cluster_shape"]["overlapping"] is False
    assert all(LEVELS[s] <= LEVELS["low"] for s in _suitability(checker).values())
    assert any("Hopkins" in caveat for caveat in checker.report["caveats"])


def test_high_dimensional_noise_concentrates_distances():
    df = _frame(np.random.default_rng(42).uniform(size=(500, 50)))
    _, results = _run(df)
    assert results["dimensionality"]["distances_concentrated"] is True


def test_very_large_n_rules_out_quadratic_methods():
    df = _frame(datasets.make_blobs(60_000, centers=3, random_state=0)[0])
    checker, results = _run(df)
    assert results["sample_size"]["very_large_n"] is True
    suitability = _suitability(checker)
    assert suitability["AgglomerativeClustering"] == "unsuitable"
    assert suitability["SpectralClustering"] == "unsuitable"
    assert suitability["MiniBatchKMeans"] == "high"


def test_single_algorithm_reports_status_reason_and_advice():
    _, results = _run(_moons(), algorithm="KMeans")
    assert results["status"] == "unsuitable"
    assert "non-convex" in results["reason"]
    assert any("n_clusters (the data suggests about 2)" in note for note in results["advice"])


def test_algorithm_aliases_are_accepted():
    checker, results = _run(_anisotropic(), algorithm="GMM")
    assert checker.algorithm == "GaussianMixture"
    assert results["status"] == "suitable"


def test_unknown_algorithm_is_reported(capsys):
    _, results = _run(_blobs(), algorithm="Birch")
    assert results["status"] == "unknown"
    assert "ClusteringAssumptionsChecker" in capsys.readouterr().out


def test_recommend_models_runs_checks_when_needed():
    checker = ClusteringAssumptionsChecker(_blobs())
    recommendations = checker.recommend_models()
    assert recommendations[0]["suitability"] == "high"
    assert "cluster_shape" in checker.results


def test_too_few_rows_skips_geometry_rather_than_passing():
    checker, results = _run(_blobs().head(10))
    assert results["cluster_shape"]["non_globular"] is None
    assert results["outliers"]["has_noise"] is None
    assert all(s != "high" for s in _suitability(checker).values())


def test_mixed_types_are_flagged_and_excluded_from_geometry():
    df = _moons()
    df["segment"] = np.where(df["x0"] > 0, "east", "west")
    _, results = _run(df)
    assert results["feature_types"]["mixed_types"] is True
    assert results["feature_types"]["non_numeric_features"] == ["segment"]
    assert results["cluster_shape"]["shape"] == "non_convex"


def test_duplicates_are_flagged():
    df = _blobs()
    df = pd.concat([df, df.head(50)], ignore_index=True)
    _, results = _run(df)
    assert results["duplicates"]["has_duplicates"] is True


def test_heavily_repeated_values_are_flagged():
    df = _blobs()
    df["visits"] = np.random.default_rng(0).choice([1, 1, 1, 2, 3, 5], size=len(df))
    checker, results = _run(df)
    assert results["tied_values"]["has_tied_values"] is True
    assert list(results["tied_values"]["tied_columns"]) == ["visits"]
    dbscan = next(r for r in checker.report["recommendations"] if r["algorithm"] == "DBSCAN")
    assert any("Repeated values in visits" in note for note in dbscan["notes"])


def test_continuous_data_has_no_tied_values():
    _, results = _run(_blobs())
    assert results["tied_values"]["has_tied_values"] is False


def test_scale_disparity_adds_standardisation_advice():
    df = _blobs()
    df["x1"] *= 1000
    checker, results = _run(df)
    assert verdict(results["scaling_issues"], "scaling_needed") is True
    kmeans = next(r for r in checker.report["recommendations"] if r["algorithm"] == "KMeans")
    assert any("Standardise" in note for note in kmeans["notes"])


def test_scale_disparity_is_found_despite_missing_values():
    df = _blobs()
    df["x1"] *= 1000
    df.loc[:9, "x0"] = np.nan
    _, results = _run(df)
    assert verdict(results["scaling_issues"], "scaling_needed") is True


def test_checker_does_not_modify_caller_df():
    df = _blobs()
    df.loc[0, "x0"] = np.nan
    original = df.copy()
    _run(df)
    pd.testing.assert_frame_equal(df, original)


def test_report_exports_to_json():
    checker, _ = _run(_blobs())
    report = json.loads(checker.export_report("json"))
    assert {"assumptions", "recommendations", "caveats"} <= report.keys()
