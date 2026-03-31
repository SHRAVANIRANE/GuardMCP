import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.alignment.cosine_baseline import CosineBaseline
from src.alignment.directional import DirectionalAlignment
from src.data.loader import get_all_cases
from src.embeddings.embedder import Embedder
from src.evaluation.evaluator import Evaluator
from src.evaluation.metrics import compute_metrics

RESULTS_DIR = ROOT_DIR / "results"
DETAILS_PATH = RESULTS_DIR / "outputs.csv"
SUMMARY_PATH = RESULTS_DIR / "results_summary.csv"
BEST_THRESHOLDS_PATH = RESULTS_DIR / "best_thresholds.csv"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run GuardMCP experiments on local and optional benchmark datasets."
    )
    parser.add_argument(
        "--include-tooltalk",
        action="store_true",
        help="Include locally converted ToolTalk aligned benchmark cases.",
    )
    parser.add_argument(
        "--include-agentdojo",
        action="store_true",
        help="Include locally converted AgentDojo injection benchmark cases.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["accuracy", "precision", "recall", "f1"],
        default="f1",
        help="Metric to maximize on the dev split when selecting thresholds.",
    )
    return parser.parse_args()


def build_candidate_thresholds(series):
    values = sorted({float(value) for value in series.dropna().tolist()})
    if not values:
        raise ValueError("cannot build thresholds from an empty score series")

    lower_margin = values[0] - 1e-6
    upper_margin = values[-1] + 1e-6
    return [lower_margin, *values, upper_margin]


def apply_method_threshold(df, method, threshold):
    applied_df = df.copy()

    if method == "directional":
        applied_df["decision"] = (applied_df["rejection_magnitude"] <= threshold).astype(int)
    elif method == "cosine":
        applied_df["decision"] = (applied_df["cosine_similarity"] >= threshold).astype(int)
    else:
        raise ValueError(f"unsupported method: {method}")

    return applied_df


def sweep_method_thresholds(df, method):
    score_column = "rejection_magnitude" if method == "directional" else "cosine_similarity"
    thresholds = build_candidate_thresholds(df[score_column])
    summary_rows = []

    for threshold in thresholds:
        threshold_df = apply_method_threshold(df, method, threshold)
        metrics = compute_metrics(threshold_df, "decision")
        summary_rows.append(
            {
                "stage": "dev_sweep",
                "split": "dev",
                "method": method,
                "threshold": threshold,
                **metrics,
            }
        )

    return pd.DataFrame(summary_rows)


def select_best_threshold(summary_df, method, selection_metric):
    method_df = summary_df[summary_df["method"] == method].copy()
    if method_df.empty:
        raise ValueError(f"no threshold sweep rows found for method {method}")

    method_df["safety_tiebreak"] = (
        -method_df["threshold"] if method == "directional" else method_df["threshold"]
    )
    ranked_df = method_df.sort_values(
        by=[selection_metric, "accuracy", "precision", "recall", "safety_tiebreak"],
        ascending=[False, False, False, False, False],
    )
    return ranked_df.iloc[0]


def summarize_test_metrics(df, method, threshold, selection_metric):
    method_df = apply_method_threshold(df, method, threshold)
    metrics = compute_metrics(method_df, "decision")
    return {
        "stage": "test_best",
        "split": "test",
        "method": method,
        "threshold": threshold,
        "selected_on": "dev",
        "selection_metric": selection_metric,
        **metrics,
    }


def main():
    args = parse_args()
    all_cases = get_all_cases(
        include_tooltalk=args.include_tooltalk,
        include_agentdojo=args.include_agentdojo,
        return_records=True,
    )
    embedder = Embedder()
    directional = DirectionalAlignment(epsilon=0.0)
    cosine = CosineBaseline(threshold=0.0)
    evaluator = Evaluator(embedder, directional, cosine)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    scored_df = evaluator.score(all_cases)

    dev_df = scored_df[scored_df["split"] == "dev"].copy()
    test_df = scored_df[scored_df["split"] == "test"].copy()

    print(
        "\nEvaluation splits -> "
        f"dev: {len(dev_df)} rows, "
        f"test: {len(test_df)} rows"
    )

    if dev_df.empty or test_df.empty:
        raise ValueError("dev and test splits must both be non-empty")

    dev_directional_summary = sweep_method_thresholds(dev_df, "directional")
    dev_cosine_summary = sweep_method_thresholds(dev_df, "cosine")
    dev_summary = pd.concat([dev_directional_summary, dev_cosine_summary], ignore_index=True)

    best_directional = select_best_threshold(dev_summary, "directional", args.selection_metric)
    best_cosine = select_best_threshold(dev_summary, "cosine", args.selection_metric)

    test_summary_rows = [
        summarize_test_metrics(
            test_df,
            method="directional",
            threshold=float(best_directional["threshold"]),
            selection_metric=args.selection_metric,
        ),
        summarize_test_metrics(
            test_df,
            method="cosine",
            threshold=float(best_cosine["threshold"]),
            selection_metric=args.selection_metric,
        ),
    ]
    test_summary = pd.DataFrame(test_summary_rows)

    final_outputs_df = evaluator.apply_thresholds(
        scored_df,
        directional_threshold=float(best_directional["threshold"]),
        cosine_threshold=float(best_cosine["threshold"]),
    )
    final_outputs_df["directional_selected_on"] = "dev"
    final_outputs_df["cosine_selected_on"] = "dev"

    results_summary = pd.concat([dev_summary, test_summary], ignore_index=True)
    best_thresholds_df = pd.DataFrame(
        [
            {
                "method": "directional",
                "threshold": float(best_directional["threshold"]),
                "dev_selection_metric": float(best_directional[args.selection_metric]),
                "selection_metric": args.selection_metric,
            },
            {
                "method": "cosine",
                "threshold": float(best_cosine["threshold"]),
                "dev_selection_metric": float(best_cosine[args.selection_metric]),
                "selection_metric": args.selection_metric,
            },
        ]
    )

    results_summary.to_csv(SUMMARY_PATH, index=False)
    best_thresholds_df.to_csv(BEST_THRESHOLDS_PATH, index=False)
    final_outputs_df.to_csv(DETAILS_PATH, index=False)

    print("\nBest thresholds selected on dev split:")
    print(
        f"Directional -> threshold: {float(best_directional['threshold']):.6f}, "
        f"{args.selection_metric}: {float(best_directional[args.selection_metric]):.3f}"
    )
    print(
        f"Cosine      -> threshold: {float(best_cosine['threshold']):.6f}, "
        f"{args.selection_metric}: {float(best_cosine[args.selection_metric]):.3f}"
    )

    print("\nFinal test metrics with dev-selected thresholds:")
    for row in test_summary_rows:
        print(
            f"{row['method'].title():<11} -> "
            f"Accuracy: {row['accuracy']:.2f}, "
            f"Precision: {row['precision']:.2f}, "
            f"Recall: {row['recall']:.2f}, "
            f"F1: {row['f1']:.2f}"
        )

    print(f"\nSaved summary results to {SUMMARY_PATH}")
    print(f"Saved best thresholds to {BEST_THRESHOLDS_PATH}")
    print(f"Saved detailed results to {DETAILS_PATH}")


if __name__ == "__main__":
    main()
