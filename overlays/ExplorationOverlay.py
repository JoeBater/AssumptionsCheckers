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

class ExaminationOverlay:
    def __init__(self, df, target, visualize=False):
        self.df = df
        self.target = target
        self.visualize = visualize
        self.X = df.drop(columns=[target])
        self.y = df[target]

    # =======================================
    # Individual checks
    # =======================================
    def check_linearity(self):
        
        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        reset_test = linear_reset(model, power=2, use_f=True)

        if self.visualize:
            sm.graphics.plot_partregress_grid(model)
            plt.suptitle("Partial Residuals")
            plt.show()

        return {
            "linearity": reset_test.pvalue >= 0.05,
            "p_value": reset_test.pvalue,
            "notes": "Linearity assumption holds." if reset_test.pvalue >= 0.05 else "Non-linearity detected."
        }

    def check_residual_normality(self):
        X_const = sm.add_constant(self.X)
        model = sm.OLS(self.y, X_const).fit()
        stat, p_value = shapiro(model.resid)

        if self.visualize:
            sm.qqplot(model.resid, line='s')
            plt.title("Q-Q Plot of Residuals")
            plt.show()

        return {
            "normality": p_value >= 0.05,
            "p_value": p_value,
            "notes": "Residuals appear normal." if p_value >= 0.05 else "Residuals deviate from normality."
        }

    def check_scaling_need(self, threshold=10):
        stds = self.X.std()
        ratio = stds.max() / stds.min()

        return {
            "scaling_needed": ratio > threshold,
            "std_ratio": ratio,
            "notes": "Scaling recommended." if ratio > threshold else "Feature scales are aligned."
        }
