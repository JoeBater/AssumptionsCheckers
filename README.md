# 🧠 AssumptionsCheckers

AssumptionsCheckers is a modular toolkit for checking whether your data suits the machine-learning models you plan to use, and for catching data-integrity problems before modelling. It's designed for diagnostic clarity, operational realism, and stakeholder transparency: every check returns a plain-language note and a recommendation alongside the numbers.

## 🔍 What it does

- ✅ **Regression diagnostics:** linearity (RESET test), homoscedasticity (Breusch–Pagan), multicollinearity (VIF with a suggested drop set), residual normality (Shapiro–Wilk), scaling, tree suitability
- ✅ **Classification diagnostics:** class imbalance, multicollinearity, scaling, feature redundancy (correlation and mutual information), class separability (PCA or t-SNE projection)
- ✅ **Clustering diagnostics (no target):** cluster shape (globular, ellipsoidal or non-convex), unequal cluster spread, elongation, density variation, noise and outliers, heavily repeated values, clustering tendency (Hopkins), distance concentration in high dimensions, scale disparity, duplicates, sample size and mixed data types
- ✅ **Model recommendations:** name an algorithm to check its specific assumptions, or leave it out to run every check and get a ranked list of candidate models
- ✅ **Data integrity checks:** missingness (with MCAR/MAR hints), whitespace and null-like tokens, constant and near-constant columns, high cardinality, duplicate rows, embedded delimited rows
- ✅ **Exploratory checks:** missing values, outliers (z-score, IQR or modified z-score, chosen per column), with suggested cleaning code

Checks that can't run (for example, because too many rows are incomplete) report `skipped` rather than a pass or fail, and recommendations treat them as unknown.

## 🚀 Getting started

```bash
git clone https://github.com/JoeBater/AssumptionsCheckers.git
cd AssumptionsCheckers
pip install -e ".[dev]"
```

`pip install -r requirements.txt` still works if you prefer the pinned environment.

```python
from assumptions import RegressionAssumptionsChecker

checker = RegressionAssumptionsChecker(df, target="price", algorithm="Ridge")
results = checker.check_assumptions()
checker.report_assumptions()
```

Leave out `algorithm` to run every check and rank candidate models. Supported algorithms:

| Checker | Algorithms |
|---|---|
| `RegressionAssumptionsChecker` | `LinearRegression`, `Ridge`, `Lasso`, `ElasticNet`, `SVR`, `KNeighborsRegressor`, `DecisionTreeRegressor`, `RandomForestRegressor`, `GradientBoostingRegressor` |
| `ClassificationAssumptionsChecker` | `LogisticRegression`, `SVC`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`, `GradientBoostingClassifier` |
| `ClusteringAssumptionsChecker` | `KMeans`, `MiniBatchKMeans`, `AgglomerativeClustering`, `DBSCAN`, `OPTICS`, `GaussianMixture`, `SpectralClustering` |

For clustering there is no target, so every column is treated as a feature:

```python
from assumptions import ClusteringAssumptionsChecker

checker = ClusteringAssumptionsChecker(df, algorithm="KMeans")  # aliases like "GMM" also work
checker.check_assumptions()      # status, reason and advice for K-Means
checker.report_assumptions()

ClusteringAssumptionsChecker(df).recommend_models()  # rank every algorithm
```

The checker has no labels to go on, so it infers cluster structure from the data. It finds density-connected regions, then checks whether K-Means or a Gaussian mixture reproduces them. It also fits a Gaussian mixture chosen by BIC to compare the clusters' spread, elongation and density. `clustering_examples.ipynb` shows it on blobs, moons, circles and other datasets whose geometry is known. `notebooks/clustering_examples.ipynb` runs it on three UCI datasets (Online Retail, Human Activity Recognition and Obesity Levels).

For data integrity:

```python
from assumptions import DataIntegrityChecker

checker = DataIntegrityChecker(df)
checker.run_all_checks()
print(checker.report())
print("\n".join(checker.recommended_actions()))
```

The checkers never modify the DataFrame you pass in; they work on a copy.

## 📦 Structure

```
AssumptionsCheckers/
├── assumptions/            # Checkers: the entry points you call
│   ├── RegressionAssumptionsChecker.py
│   ├── ClassificationAssumptionsChecker.py
│   ├── ClusteringAssumptionsChecker.py
│   ├── DataIntegrityChecker.py
│   ├── ExploratoryDataChecker.py
│   └── _common.py          # Shared checker base class and reporting
├── overlays/               # Individual diagnostic checks used by the checkers
├── notebooks/              # Further example notebooks
├── tests/                  # pytest suite
├── *_examples.ipynb        # Worked examples for each checker
└── pyproject.toml
```

## 🧪 Tests

```bash
pytest
```

## 📓 Notebook hygiene

The repo includes a `.gitattributes` entry for [nbstripout](https://github.com/kynan/nbstripout). Run this once per clone to enable it; example outputs are kept, and only execution counts and kernel metadata are stripped:

```bash
nbstripout --install --keep-output --extra-keys "metadata.kernelspec metadata.language_info.version"
```

## 🧭 Roadmap

- [x] Regression diagnostics
- [x] Classification diagnostics
- [x] Data integrity checks
- [ ] Distributional diagnostics: skewness, outlier and drift checks (`overlays/DistributionalDiagnosticsOverlay.py`)
- [ ] Column confidence and trust scoring (`overlays/ConfidenceTrustOverlay.py`)
- [ ] Text data assumptions (`assumptions/TextAssumptionsChecker.py`)
- [x] Clustering diagnostics: KMeans, MiniBatchKMeans, Agglomerative, DBSCAN, OPTICS, GMM, Spectral
- [ ] More clustering algorithms: HDBSCAN, K-Medoids
- [ ] Time-series support
- [ ] Interactive dashboard (streamlit or gradio)

## 🤝 Contributing

Pull requests welcome! Please submit modular overlays with clear diagnostics and stakeholder-aligned reporting, and add tests under `tests/`.
