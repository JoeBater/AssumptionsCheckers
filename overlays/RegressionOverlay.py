import statsmodels.api as sm
from statsmodels.stats.diagnostic import linear_reset
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro

from ._common import DEFAULT_MISSING_TOKENS, MissingnessAwareOverlay, split_numeric, vif_drop_set


class RegressionOverlay(MissingnessAwareOverlay):
    def __init__(self, df, target, visualize=False, custom_missing_values=None, missingness_threshold=0.5, suppress_missing_warnings=False):
        super().__init__(
            df, target, visualize,  # visualize not used currently
            missingness_threshold=missingness_threshold,
            suppress_missing_warnings=suppress_missing_warnings,
            missing_tokens=custom_missing_values or DEFAULT_MISSING_TOKENS,
        )

    def _fit_ols(self):
        """Fit OLS on numeric predictors; OLS cannot take non-numeric columns."""
        df_to_use = self.working_df
        X, excluded = split_numeric(df_to_use.drop(columns=[self.target]))
        model = sm.OLS(df_to_use[self.target], sm.add_constant(X)).fit()
        note = f" Non-numeric columns excluded: {excluded}." if excluded else ""
        return model, note

    # =======================================
    # Individual checks
    # =======================================
    def check_linearity(self):
        if self.too_much_missing("Linearity test"):
            return {
                "linearity": None,
                "p_value": None,
                "notes": "Linearity test not performed due to excessive missing data."
            }

        model, excluded_note = self._fit_ols()
        reset_test = linear_reset(model, power=2, use_f=True)

        return {
            "linearity": reset_test.pvalue >= 0.05,
            "p_value": reset_test.pvalue,
            "notes": (
                "Linearity assumption holds." if reset_test.pvalue >= 0.05
                else "Non-linearity detected."
            ) + excluded_note
        }

    def check_residual_normality(self):
        if self.too_much_missing("Residual normality test"):
            return {
                "normality": None,
                "p_value": None,
                "notes": "Residual normality test not performed due to excessive missing data.",
                "data_used": "None"
            }

        model, excluded_note = self._fit_ols()
        stat, p_value = shapiro(model.resid)

        return {
            "normality": p_value >= 0.05,
            "p_value": p_value,
            "notes": (
                "Residuals appear normal." if p_value >= 0.05
                else "Residuals deviate from normality."
            ) + excluded_note,
            "data_used": self.data_used
        }

    def check_multicollinearity(self, threshold=10):
        """
        Check multicollinearity using VIF, only if 2+ predictors.
        Returns diagnostics and minimal drop set recommendation.
        """
        if self.too_much_missing("Multicollinearity check"):
            return {
                "has_multicollinearity": None,
                "details": None,
                "notes": "Multicollinearity check not performed due to excessive missing data.",
                "recommendation": None,
                "drop_set": None
            }

        X, dropped_columns = split_numeric(self.working_df.drop(columns=[self.target]))  # VIF only for numeric predictors
        if dropped_columns:
            print(f"Note: Dropped non-numeric columns for VIF calculation: {dropped_columns}")

        if X.shape[1] < 2:
            return {
                "has_multicollinearity": False,
                "details": None,
                "notes": f"Not enough predictors ({X.shape[1]}) — multicollinearity check not applicable.",
                "recommendation": None
            }

        vif_data, drop_set = vif_drop_set(X, threshold)

        recommendation = []
        if drop_set:
            recommendation.append(
                f"High multicollinearity detected. Consider removing these features: {drop_set}"
            )
            recommendation.append(
                "Alternatively, use Ridge regression to mitigate multicollinearity without dropping predictors."
            )
            if X.shape[1] >= 10:
                recommendation.append(
                    "If you suspect some predictors are irrelevant, Lasso regression can perform automatic feature selection."
                )

        return {
            "has_multicollinearity": len(drop_set) > 0,
            "details": vif_data.to_dict(orient="records"),
            "notes": (
                "Perfect or high multicollinearity detected" if drop_set
                else "No severe multicollinearity."
            ),
            "recommendation": " ".join(recommendation) if recommendation else "No action needed.",
            "drop_set": drop_set
        }

    def check_heteroscedasticity(self):
        """Check heteroscedasticity using Breusch-Pagan test."""
        if self.too_much_missing("Heteroscedasticity test"):
            return {
                "homoscedasticity": None,
                "p_value": None,
                "notes": "Heteroscedasticity test not performed due to excessive missing data."
            }

        model, excluded_note = self._fit_ols()
        lm_stat, lm_pvalue, f_stat, f_pvalue = het_breuschpagan(model.resid, model.model.exog)

        return {
            "homoscedasticity": lm_pvalue >= 0.05,
            "p_value": lm_pvalue,
            "notes": (
                "Residuals are homoscedastic." if lm_pvalue >= 0.05 else "Residuals are heteroscedastic."
            ) + excluded_note
        }
