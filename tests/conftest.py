import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The package is imported from the repo root (as the notebooks do).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

warnings.filterwarnings("ignore")


@pytest.fixture
def rng():
    return np.random.default_rng(0)


@pytest.fixture
def classification_df(rng):
    """Three independent numeric features and a binary target."""
    X = rng.normal(size=(200, 3))
    df = pd.DataFrame(X, columns=["p", "q", "r"])
    df["t"] = (X[:, 0] > 0).astype(int)
    return df


@pytest.fixture
def regression_df(rng):
    """Three independent numeric features and a linear target."""
    X = rng.normal(size=(200, 3))
    df = pd.DataFrame(X, columns=["p", "q", "r"])
    df["t"] = X @ [1.0, 2.0, 3.0] + rng.normal(scale=0.1, size=200)
    return df
