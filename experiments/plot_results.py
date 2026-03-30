from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

SUMMARY_PATH = ROOT_DIR / "results" / "results_summary.csv"
PLOTS_DIR = ROOT_DIR / "results" / "plots"
PLOT_PATH = PLOTS_DIR / "precision_recall_graph.png"


def main():
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Summary results not found at {SUMMARY_PATH}. "
            "Run experiments/run_experiments.py first."
        )

    summary_df = pd.read_csv(SUMMARY_PATH).sort_values("epsilon")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(summary_df["epsilon"], summary_df["precision"], marker="o", label="Precision")
    plt.plot(summary_df["epsilon"], summary_df["recall"], marker="o", label="Recall")

    plt.xlabel("Epsilon")
    plt.ylabel("Score")
    plt.ylim(0, 1.05)
    plt.title("Precision and Recall vs Epsilon")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
