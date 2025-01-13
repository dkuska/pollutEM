import os
import matplotlib.pyplot as plt
import numpy as np


def generate_visualizations(metrics_df, output_dir, metric="f1_score"):
    """
    Generate visualizations for F1-Score, Precision, or Recall.

    Parameters:
        metrics_df (DataFrame): Data containing the metrics.
        output_dir (str): Directory where the output plot will be saved.
        metric (str): The metric to plot. Options: 'f1_score', 'precision', 'recall'.
    """
    if metric not in ["f1_score", "precision", "recall"]:
        raise ValueError("Invalid metric. Choose 'f1_score', 'precision', or 'recall'.")

    plt.figure(figsize=(14, 10))
    markers = {"XGBoost": "o", "ChatGPT": "^"}
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for pollution_type in metrics_df["pollution_type"].unique():
        type_mask = metrics_df["pollution_type"] == pollution_type

        for params_str in metrics_df[type_mask]["params"].unique():
            for matcher in metrics_df["matcher"].unique():
                mask = (
                    (metrics_df["pollution_type"] == pollution_type)
                    & (metrics_df["matcher"] == matcher)
                    & (metrics_df["params"] == params_str)
                )
                subset = metrics_df[mask]

                if len(subset) == 0:
                    continue

                params_display = params_str.replace("{", "").replace("}", "").replace("'", "")

                stats = (
                    subset.groupby("number_of_columns")[metric].agg(["mean", "std"]).reset_index()
                )
                counts = subset.groupby("number_of_columns").size()
                stats["stderr"] = stats["std"] / np.sqrt(counts)

                plt.errorbar(
                    stats["number_of_columns"],
                    stats["mean"],
                    yerr=stats["stderr"],
                    label=f"{pollution_type} ({params_display}) - {matcher}",
                    marker=markers[matcher],
                    capsize=5,
                    capthick=1,
                    markersize=8,
                    alpha=0.7,
                    color=colors[color_idx % len(colors)],
                )

            color_idx += 1

    plt.xlabel("Number of Affected Columns")
    plt.ylabel(f"Average {metric.replace('_', ' ').title()}")
    plt.title(
        f"Average {metric.replace('_', ' ').title()} vs Number of Affected Columns by Pollution Type, Parameters, and Matcher"
    )
    plt.legend(
        title="Pollution Type - Matcher",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    filename = f"{metric}_errorbar.png"
    plt.savefig(os.path.join(output_dir, filename), bbox_inches="tight")
