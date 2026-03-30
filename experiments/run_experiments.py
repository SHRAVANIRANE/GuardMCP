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


def main():
    test_cases = get_all_cases()
    embedder = Embedder()
    cosine = CosineBaseline()
    epsilons = [0.2, 0.5, 0.7, 0.8, 0.9]

    results_summary = []
    detailed_results = []

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for eps in epsilons:
        directional = DirectionalAlignment(epsilon=eps)
        evaluator = Evaluator(embedder, directional, cosine)

        df = evaluator.run(test_cases).copy()
        df["epsilon"] = eps
        detailed_results.append(df)

        accuracy, precision, recall = compute_metrics(df)

        print(f"\nEpsilon: {eps}")
        print(
            f"Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}"
        )

        results_summary.append(
            {
                "epsilon": eps,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
            }
        )

    pd.DataFrame(results_summary).to_csv(SUMMARY_PATH, index=False)
    pd.concat(detailed_results, ignore_index=True).to_csv(DETAILS_PATH, index=False)

    print(f"\nSaved summary results to {SUMMARY_PATH}")
    print(f"Saved detailed results to {DETAILS_PATH}")


if __name__ == "__main__":
    main()
