import os
import matplotlib.pyplot as plt
import numpy as np


def generate_visualizations(f1_df, output_dir):
    plt.figure(figsize=(12, 8))
    markers = {"XGBoost": "o", "ChatGPT": "^"}

    for pollution_type in f1_df["pollution_type"].unique():
        for matcher in f1_df["matcher"].unique():
            mask = (f1_df["pollution_type"] == pollution_type) & (f1_df["matcher"] == matcher)
            subset = f1_df[mask]

            # Group by number of columns and calculate statistics
            stats = (
                subset.groupby("number_of_columns")["f1_score"].agg(["mean", "std"]).reset_index()
            )

            # Calculate standard error (std/sqrt(n))
            counts = subset.groupby("number_of_columns").size()
            stats["stderr"] = stats["std"] / np.sqrt(counts)

            plt.errorbar(
                stats["number_of_columns"],
                stats["mean"],
                yerr=stats["stderr"],
                label=f"{pollution_type} - {matcher.split('.')[-1].replace('>', '')}",
                marker=markers[matcher],
                capsize=5,
                capthick=1,
                markersize=8,
                alpha=0.7,
            )

    plt.xlabel("Number of Affected Columns")
    plt.ylabel("Average F1 Score")
    plt.title("Average F1 Score vs Number of Affected Columns by Pollution Type and Matcher")
    plt.legend(title="Pollution Type - Matcher", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "f1_score_errorbar.png"), bbox_inches="tight")
