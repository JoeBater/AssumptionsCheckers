NO_ALGORITHM_NOTE = (
    "ℹ️ No algorithm specified — running all checks and ranking candidate models. "
    "Pass algorithm='...' to check a specific model (see help_algorithms())."
)

# Result keys holding a check's verdict: True/False, or None when the check was skipped
VERDICT_KEYS = {
    "linearity", "homoscedasticity", "normality", "has_multicollinearity",
    "scaling_needed", "tree_suitability", "separable", "is_imbalanced",
}


def verdict(result, key):
    """Return a check's True/False verdict, or None if the check was skipped."""
    value = result.get(key) if isinstance(result, dict) else None
    # Overlays often return numpy bools, which fail identity checks like `is False`
    return None if value is None else bool(value)


def print_check_result(result):
    """Print one check's result dict, showing skipped verdicts as 'skipped'."""
    for subkey, value in result.items():
        # Skip redundant missingness messages from individual checks
        if subkey in ["notes", "recommendation"] and "missing" in str(value).lower():
            continue
        if value is None:
            if subkey in VERDICT_KEYS:
                print(f"{subkey}: skipped")
            continue
        print(f"{subkey}: {value}")
