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

    summary_df = pd.read_csv(SUMMARY_PATH).sort_values(["method", "threshold"])
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    metrics = ["accuracy", "precision", "recall"]

    for axis, metric_name in zip(axes, metrics):
        for method in summary_df["method"].unique():
            method_df = summary_df[summary_df["method"] == method]
            axis.plot(
                method_df["threshold"],
                method_df[metric_name],
                marker="o",
                label=method.title(),
            )

        axis.set_title(metric_name.title())
        axis.set_xlabel("Threshold")
        axis.set_ylim(0, 1.05)
        axis.grid(alpha=0.3)

    axes[0].set_ylabel("Score")
    axes[-1].legend()
    fig.suptitle("Directional vs Cosine Metrics Across Thresholds")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
