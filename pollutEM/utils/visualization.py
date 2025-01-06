import os
import matplotlib.pyplot as plt


def generate_visualizations(f1_df, output_dir):
    # Create scatter plot
    plt.figure(figsize=(12, 8))

    # Create scatter plot for each pollution type with different colors
    for pollution_type in f1_df["pollution_type"].unique():
        mask = f1_df["pollution_type"] == pollution_type
        plt.scatter(
            f1_df.loc[mask, "number_of_columns"],
            f1_df.loc[mask, "f1_score"],
            label=pollution_type,
            alpha=0.7,
        )

    # Customize the plot
    plt.xlabel("Number of Columns")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Number of Columns by Pollution Type")
    plt.legend(title="Pollution Type")
    plt.grid(True, linestyle="--", alpha=0.7)

    # Save the plot
    plt.savefig(os.path.join(output_dir, "f1_score_scatter.png"))
    plt.close()
