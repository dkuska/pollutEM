import os
import subprocess
import pandas as pd
import click


def generate_configurations(dataset_path, master_config_path, output_dir, samples_per_size):
    print("Generating configurations...")
    subprocess.run(
        [
            "python",
            "configuration_generator.py",
            "--dataset_path",
            dataset_path,
            "--master_config_path",
            master_config_path,
            "--output_dir",
            output_dir,
            "--samples_per_size",
            str(samples_per_size),
        ]
    )


def apply_pollution_to_dataset(original_dataset, config_path, polluted_output):
    print(f"Applying pollution using config {config_path}...")
    subprocess.run(
        [
            "python",
            "polluter.py",
            "--input-file",
            original_dataset,
            "--config-file",
            config_path,
            "--output-file",
            polluted_output,
        ]
    )


def evaluate_model(model_path, original_dataset, polluted_dataset, test_split, mode, results_dir):
    print("Evaluating model...")
    subprocess.run(
        [
            "python",
            "evaluator.py",
            "--original",
            original_dataset,
            "--polluted",
            polluted_dataset,
            "--test_split",
            test_split,
            "--model",
            model_path,
            "--mode",
            mode,
            "--results-dir",
            results_dir,
        ]
    )


def generate_visualizations(results_dir):
    print("Generating visualizations...")
    f1_df = pd.read_csv(os.path.join(results_dir, "evaluation_results.csv"))
    import matplotlib.pyplot as plt

    f1_df.plot(kind="box", figsize=(10, 6))
    plt.title("F1 Score Distribution")
    plt.savefig(os.path.join(results_dir, "f1_score_boxplot.png"))
    plt.close()


@click.command()
@click.option("--dataset-path", type=str, required=True, help="Path to the original dataset")
@click.option(
    "--master-config-path", type=str, required=True, help="Path to the master configuration file"
)
@click.option(
    "--config-dir", type=str, required=True, help="Directory to save individual configurations"
)
@click.option(
    "--samples-per-size",
    type=int,
    default=5,
    help="Number of random combinations to generate for each combination size",
)
@click.option("--model-path", type=str, required=True, help="Path to the trained model")
@click.option("--test-split-path", type=str, required=True, help="Path to the test split file")
@click.option(
    "--results-dir",
    type=str,
    required=True,
    help="Directory to save evaluation results and reports",
)
@click.option(
    "--mode",
    type=click.Choice(["original", "polluted", "mixed"], case_sensitive=False),
    required=True,
    help="Mode for feature generation",
)
def run_pipeline(
    dataset_path,
    master_config_path,
    config_dir,
    samples_per_size,
    model_path,
    test_split_path,
    results_dir,
    mode,
):
    # Step 1: Generate individual configurations
    generate_configurations(dataset_path, master_config_path, config_dir, samples_per_size)

    # Step 2: Apply pollution to dataset for each configuration
    for config_file in os.listdir(config_dir):
        if config_file.endswith(".yaml"):
            config_path = os.path.join(config_dir, config_file)
            polluted_output = os.path.join(
                results_dir, f"polluted_{config_file.replace('.yaml', '.csv')}"
            )
            apply_pollution_to_dataset(dataset_path, config_path, polluted_output)

            # Step 3: Evaluate the model on the polluted dataset
            evaluate_model(
                model_path, dataset_path, polluted_output, test_split_path, mode, results_dir
            )

    # Step 4: Create visualizations
    generate_visualizations(results_dir)


if __name__ == "__main__":
    run_pipeline()
