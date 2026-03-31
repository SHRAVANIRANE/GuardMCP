import pandas as pd

from src.evaluation.metrics import compute_metrics


def _normalize_group_values(df, group_column, include_missing):
    normalized_df = df.copy()

    if include_missing:
        normalized_df[group_column] = (
            normalized_df[group_column]
            .fillna("unspecified")
            .replace("", "unspecified")
        )
        return normalized_df

    return normalized_df[
        normalized_df[group_column].notna()
        & (normalized_df[group_column].astype(str).str.strip() != "")
    ].copy()


def build_grouped_report(df, group_column, method, decision_column, include_missing=False):
    if group_column not in df.columns:
        raise ValueError(f"{group_column} is not present in the dataframe")

    grouped_df = _normalize_group_values(df, group_column, include_missing=include_missing)
    rows = []

    for group_value, slice_df in grouped_df.groupby(group_column, dropna=False):
        metrics = compute_metrics(slice_df, decision_column)
        rows.append(
            {
                "group_by": group_column,
                "group_value": group_value,
                "method": method,
                **metrics,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "group_by",
                "group_value",
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "tp",
                "tn",
                "fp",
                "fn",
                "positive_support",
                "negative_support",
                "allow_rate",
                "block_rate",
                "support",
            ]
        )

    report_df = pd.DataFrame(rows)
    return report_df.sort_values(["support", "group_value"], ascending=[False, True]).reset_index(drop=True)


def build_reporting_bundle(test_df):
    report_specs = {
        "source": {"group_column": "source", "include_missing": True},
        "suite": {"group_column": "suite", "include_missing": False},
        "attack_type": {"group_column": "attack_type", "include_missing": True},
    }
    method_specs = {
        "directional": "directional_decision",
        "cosine": "cosine_decision",
    }

    reports = {}
    for report_name, report_spec in report_specs.items():
        per_method_reports = []
        for method, decision_column in method_specs.items():
            per_method_reports.append(
                build_grouped_report(
                    test_df,
                    group_column=report_spec["group_column"],
                    method=method,
                    decision_column=decision_column,
                    include_missing=report_spec["include_missing"],
                )
            )
        reports[report_name] = pd.concat(per_method_reports, ignore_index=True)

    return reports
