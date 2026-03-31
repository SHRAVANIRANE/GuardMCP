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
    return parser.parse_args()


def main():
    args = parse_args()
    test_cases = get_all_cases(
        include_tooltalk=args.include_tooltalk,
        include_agentdojo=args.include_agentdojo,
    )
    embedder = Embedder()
    thresholds = [0.2, 0.5, 0.7, 0.8, 0.9]

    results_summary = []
    detailed_results = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for threshold in thresholds:
        directional = DirectionalAlignment(epsilon=threshold)
        cosine = CosineBaseline(threshold=threshold)
        evaluator = Evaluator(embedder, directional, cosine)

        df = evaluator.run(test_cases).copy()
        df["threshold"] = threshold
        detailed_results.append(df)

        directional_metrics = compute_metrics(df, "directional_decision")
        cosine_metrics = compute_metrics(df, "cosine_decision")

        print(f"\nThreshold: {threshold}")
        print(
            "Directional -> "
            f"Accuracy: {directional_metrics['accuracy']:.2f}, "
            f"Precision: {directional_metrics['precision']:.2f}, "
            f"Recall: {directional_metrics['recall']:.2f}"
        )
        print(
            "Cosine      -> "
            f"Accuracy: {cosine_metrics['accuracy']:.2f}, "
            f"Precision: {cosine_metrics['precision']:.2f}, "
            f"Recall: {cosine_metrics['recall']:.2f}"
        )

        for method, metrics in (
            ("directional", directional_metrics),
            ("cosine", cosine_metrics),
        ):
            results_summary.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "accuracy": metrics["accuracy"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                }
            )

    pd.DataFrame(results_summary).to_csv(SUMMARY_PATH, index=False)
    pd.concat(detailed_results, ignore_index=True).to_csv(DETAILS_PATH, index=False)

    print(f"\nSaved summary results to {SUMMARY_PATH}")
    print(f"Saved detailed results to {DETAILS_PATH}")


if __name__ == "__main__":
    main()
