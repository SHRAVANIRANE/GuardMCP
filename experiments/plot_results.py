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

    summary_df = pd.read_csv(SUMMARY_PATH)
    dev_df = summary_df[summary_df["stage"] == "dev_sweep"].sort_values(["method", "threshold"])
    best_df = summary_df[summary_df["stage"] == "test_best"].sort_values(["method"])

    if dev_df.empty:
        raise ValueError("No dev_sweep rows found in results_summary.csv.")

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    metrics = ["accuracy", "precision", "recall"]

    for axis, metric_name in zip(axes, metrics):
        for method in dev_df["method"].unique():
            method_df = dev_df[dev_df["method"] == method]
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
    if best_df.empty:
        fig.suptitle("Directional vs Cosine Metrics Across Dev Threshold Sweep")
    else:
        best_note = "; ".join(
            f"{row.method.title()} best={row.threshold:.4f}"
            for row in best_df.itertuples()
        )
        fig.suptitle(f"Directional vs Cosine Metrics Across Dev Threshold Sweep\n{best_note}")
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=200)
    plt.close()

    print(f"Saved plot to {PLOT_PATH}")


if __name__ == "__main__":
    main()
