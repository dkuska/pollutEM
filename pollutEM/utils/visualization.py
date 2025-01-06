import os
import matplotlib.pyplot as plt


def generate_visualizations(f1_df, output_dir):
    plt.figure(figsize=(12, 8))

    # Define markers for different matchers
    markers = {"XGBoost": "o", "ChatGPT": "^"}

    # Create scatter plot for each combination of pollution type and matcher
    for pollution_type in f1_df["pollution_type"].unique():
        for matcher in f1_df["matcher"].unique():
            mask = (f1_df["pollution_type"] == pollution_type) & (f1_df["matcher"] == matcher)
            plt.scatter(
                f1_df.loc[mask, "number_of_columns"],
                f1_df.loc[mask, "f1_score"],
                label=f"{pollution_type} - {matcher.split('.')[-1].replace('>', '')}",
                marker=markers[matcher],
                alpha=0.7,
            )

    plt.xlabel("Number of Columns")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Number of Columns by Pollution Type and Matcher")
    plt.legend(title="Pollution Type - Matcher", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, "f1_score_scatter.png"), bbox_inches="tight")
