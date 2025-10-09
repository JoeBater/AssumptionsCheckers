import pandas as pd

from overlays.DataIntegrityOverlay import DataIntegrityOverlay
from overlays.DistributionalDiagnosticsOverlay import DistributionalDiagnosticsOverlay
from overlays.ConfidenceTrustOverlay import ConfidenceTrustOverlay

class DataIntegrityChecker:
    def __init__(self, df, custom_missing_values=None, visualize=False):
        self.df = df
        self.custom_missing_values = custom_missing_values
        self.data_integrity_overlay = DataIntegrityOverlay(df, self.custom_missing_values, visualize)
        self.distributional_diagnostics_overlay = DistributionalDiagnosticsOverlay(df, visualize)
        self.confidence_trust_overlay = ConfidenceTrustOverlay(df, visualize)
        self.results = {}

    def run_all_checks(self):

        self.results["missingness"] = self.data_integrity_overlay.check_missingness(
            custom_missing_values=self.custom_missing_values
        )

        self.results["white_space"] = self.data_integrity_overlay.check_whitespace()

        self.results["constancy"] = self.data_integrity_overlay.check_constancy()

        self.results["cardinality"] = self.data_integrity_overlay.check_high_cardinality()

        self.results["duplicated_rows"] = self.data_integrity_overlay.check_duplicate_rows()

        self.results["embedded_rows"] = self.data_integrity_overlay.check_embedded_rows_by_length()
    
        # self.report = {
        #     "structural_integrity_checks": self.data_integrity_overlay.run_checks(),
        #     "distributional_diagnostics": self.distributional_diagnostics_overlay.run_checks(),
        #     "confidence_trust": self.confidence_trust_overlay.run_checks(),
        # }

    def run_structural_integrity_checks(self):
        pass

    def run_distributional_diagnostics_checks(self):
        pass

    def run_confidence_trust_checks(self):
        pass

    def report(self):
        if not self.results:
            return "No diagnostics have been run yet."

        status_icon = {
            "ok": "✅",
            "warning": "⚠️",
            "critical": "❌"
        }

        summary_lines = []
        for check_name, result in self.results.items():
            icon = status_icon.get(result.get("status", "warning"))
            base_report = result.get("report", "No summary available.")

            # Add column-level details for missingness
            if check_name == "missingness":
                col_details = result.get("high_missing_columns", {})
                if col_details:
                    top_cols = sorted(col_details.items(), key=lambda x: -x[1])[:5]
                    col_report = ", ".join([f"`{col}` ({pct*100:.1f}%)" for col, pct in top_cols])
                    base_report += f" Top columns: {col_report}."

                row_details = result.get("high_missing_rows", {})
                if row_details:
                    base_report += f" {len(row_details)} rows exceed missingness threshold."

            elif check_name == "white_space":
                ws_details = result.get("whitespace_issues", {})
                if ws_details:
                    top_ws = sorted(ws_details.items(), key=lambda x: -x[1]["total"])[:5]
                    col_report = ", ".join([
                        f"`{col}` (null-like: {v['null_like']}, lead: {v['leading_space']}, trail: {v['trailing_space']})"
                        for col, v in top_ws
                    ])
                    base_report += f" Top columns: {col_report}."

            elif check_name == "constancy":
                const_cols = result.get("constant_columns", [])
                near_const = result.get("near_constant_columns", {})
                if const_cols or near_const:
                    details = []
                    if const_cols:
                        details.append(f"{len(const_cols)} constant columns: " + ", ".join([f"`{col}`" for col in const_cols[:5]]))
                    if near_const:
                        sorted_near = sorted(near_const.items(), key=lambda x: -x[1])[:5]
                        details.append("Near-constant columns: " + ", ".join([f"`{col}` ({ratio*100:.1f}%)" for col, ratio in sorted_near]))
                    base_report += " " + "; ".join(details)
                else:
                    base_report = "No constant or near-constant columns found."

            elif check_name == "high_cardinality":
                hc = result.get("high_cardinality", {})
                if hc:
                    top_hc = sorted(hc.items(), key=lambda x: -x[1]["unique_values"])[:5]
                    col_report = ", ".join([
                        f"`{col}` ({v['unique_values']} unique, {v['unique_ratio']*100:.1f}% unique)"
                        for col, v in top_hc
                    ])
                    base_report += f" Top columns: {col_report}."
                else:
                    base_report = "No high-cardinality columns found."

            elif check_name == "duplicates":
                dup_count = result.get("duplicate_count", 0)
                dup_preview = result.get("duplicate_indices", [])
                if dup_count > 0:
                    preview = f" Preview: indices {dup_preview}" if dup_preview else ""
                    base_report += f" {dup_count} duplicated rows detected.{preview}"
                else:
                    base_report = "No duplicated rows found."


            summary_lines.append(f"{icon} {check_name}: {base_report}")

        return "\n".join(summary_lines)


    def recommended_actions(self):
        if not self.results:
            return ["No diagnostics have been run yet."]

        actions = []
        for check_name, result in self.results.items():
            recs = result.get("recommended_actions", [])
            if recs:
                actions.append(f"# {check_name.upper()} recommendations:")
                actions.extend(recs)
                actions.append("")  # spacer

        return actions


    