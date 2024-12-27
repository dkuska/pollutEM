import logging
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import click


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - PIPELINE - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_configurations(dataset_path, master_config_path, output_dir, samples_per_size):
    logger.info("Generating configurations...")
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
    logger.info(f"Applying pollution using config {config_path}...")
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


def evaluate_model(
    original_dataset,
    polluted_dataset,
    train_split,
    validation_split,
    test_split,
    mode,
    results_dir,
):
    logger.info("Evaluating model...")
    subprocess.run(
        [
            "python",
            "evaluator.py",
            "--original",
            original_dataset,
            "--polluted",
            polluted_dataset,
            "--train-split",
            train_split,
            "--validation-split",
            validation_split,
            "--test-split",
            test_split,
            "--mode",
            mode,
            "--results-dir",
            results_dir,
        ]
    )


def generate_visualizations(results_dir):
    logger.info("Generating visualizations...")
    f1_df = pd.read_csv(os.path.join(results_dir, "evaluation_results.csv"))
    import matplotlib.pyplot as plt

    f1_df.plot(kind="box", figsize=(10, 6))
    plt.title("F1 Score Distribution")
    plt.savefig(os.path.join(results_dir, "f1_score_boxplot.png"))
    plt.close()


def cleanup_files(config_dir: Path, polluted_datasets_dir: Path):
    """
    Clean up temporary files and directories created during the pipeline run.

    Args:
        config_dir: Directory containing configuration files
        polluted_datasets_dir: Directory containing polluted datasets
    """
    logger.info("Starting cleanup...")

    # Remove configuration directory if it exists
    if config_dir.exists():
        logger.info(f"Removing configuration directory: {config_dir}")
        shutil.rmtree(config_dir)

    # Remove polluted datasets directory if it exists
    if polluted_datasets_dir.exists():
        logger.info(f"Removing polluted datasets directory: {polluted_datasets_dir}")
        shutil.rmtree(polluted_datasets_dir)

    logger.info("Cleanup completed")


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
@click.option("--train-split-path", type=str, required=True, help="Path to the train split file")
@click.option(
    "--validation-split-path", type=str, required=True, help="Path to the validation split file"
)
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
    train_split_path,
    validation_split_path,
    test_split_path,
    results_dir,
    mode,
):
    try:
        # Convert paths to Path objects for better handling
        config_dir = Path(config_dir)
        results_dir = Path(results_dir)
        polluted_datasets_dir = results_dir / "datasets"

        # Ensure directories exist
        config_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
        polluted_datasets_dir.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate individual configurations
        generate_configurations(
            dataset_path=dataset_path,
            master_config_path=master_config_path,
            output_dir=config_dir,
            samples_per_size=samples_per_size,
        )

        # Step 2: Apply pollution to dataset for each configuration
        for config_file in os.listdir(config_dir):
            if config_file.endswith(".yaml"):
                config_path = os.path.join(config_dir, config_file)
                polluted_output = os.path.join(
                    results_dir, "datasets", f"polluted_{config_file.replace('.yaml', '.csv')}"
                )
                apply_pollution_to_dataset(
                    original_dataset=dataset_path,
                    config_path=config_path,
                    polluted_output=polluted_output,
                )

                # Step 3: Evaluate the model on the polluted dataset
                evaluate_model(
                    original_dataset=dataset_path,
                    polluted_dataset=polluted_output,
                    train_split=train_split_path,
                    validation_split=validation_split_path,
                    test_split=test_split_path,
                    mode=mode,
                    results_dir=results_dir,
                )

        # Step 4: Create visualizations
        generate_visualizations(results_dir)

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise
    finally:
        cleanup_files(config_dir, polluted_datasets_dir)


if __name__ == "__main__":
    run_pipeline()
