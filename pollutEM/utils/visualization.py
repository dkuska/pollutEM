import os
import matplotlib.pyplot as plt
import numpy as np


def generate_visualizations(f1_df, output_dir):
    plt.figure(figsize=(14, 10))
    markers = {"XGBoost": "o", "ChatGPT": "^"}
    colors = plt.cm.tab20(np.linspace(0, 1, 20))
    color_idx = 0

    for pollution_type in f1_df["pollution_type"].unique():
        type_mask = f1_df["pollution_type"] == pollution_type

        for params_str in f1_df[type_mask]["params"].unique():
            for matcher in f1_df["matcher"].unique():
                mask = (
                    (f1_df["pollution_type"] == pollution_type)
                    & (f1_df["matcher"] == matcher)
                    & (f1_df["params"] == params_str)
                )
                subset = f1_df[mask]

                if len(subset) == 0:
                    continue

                params_display = params_str.replace("{", "").replace("}", "").replace("'", "")

                stats = (
                    subset.groupby("number_of_columns")["f1_score"]
                    .agg(["mean", "std"])
                    .reset_index()
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
    plt.ylabel("Average F1 Score")
    plt.title(
        "Average F1 Score vs Number of Affected Columns by Pollution Type, Parameters, and Matcher"
    )
    plt.legend(
        title="Pollution Type - Matcher",
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize="small",
    )
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_errorbar.png"), bbox_inches="tight")
