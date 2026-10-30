"""Placeholder: early exploration checks, not yet wired into any checker (see README roadmap)."""
import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from scipy.stats import shapiro

import matplotlib.pyplot as plt

class ExplorationOverlay:
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
