import click
import yaml
import os
import pandas as pd
import random
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any
import logging
import sys

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

from polluters import get_polluter
from matchers.xgboost import XGBoostMatcher


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - MAIN - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load and validate the configuration file."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration: {str(e)}")
        sys.exit(1)


def get_pollution_templates(master_config: Dict[str, Any]) -> Dict[str, Dict]:
    """Extract unique pollution types, their parameters, and applicable columns."""
    pollution_templates = {}

    for pollution in master_config["pollutions"]:
        name = pollution["name"]
        params = pollution["params"].copy()
        params.pop("indices", None)
        params.pop("probability", None)
        params["level"] = "column"

        # Store both params and applicable columns
        pollution_templates[name] = {
            "params": params,
            "applicable_columns": pollution.get("applicable_columns", []),
        }

    return pollution_templates


def create_pollution_config(name: str, params: Dict, columns: List[str]) -> Dict[str, Any]:
    """Create a pollution configuration for given columns using template parameters."""
    config = {"name": name, "params": params.copy()}
    config["params"]["indices"] = columns
    return {"pollutions": [config]}


def generate_configs(name: str, template: Dict, all_columns: List[str], samples_per_size: int = 5):
    """Generate random samples of column combinations for each size."""
    # Get applicable columns that exist in the dataset
    applicable_columns = [col for col in template["applicable_columns"] if col in all_columns]

    if not applicable_columns:
        logger.info(f"Warning: No applicable columns found for {name}")
        return

    logger.info(f"  Applicable columns: {', '.join(applicable_columns)}")

    # Generate configurations for different combination sizes
    for r in range(1, len(applicable_columns) + 1):
        # Get all possible combinations of size r
        all_combinations = list(combinations(applicable_columns, r))
        # Randomly sample from them
        n_samples = min(samples_per_size, len(all_combinations))
        if n_samples > 0:
            selected = random.sample(all_combinations, n_samples)

            # Generate and save configs for selected combinations
            for cols in selected:
                config = create_pollution_config(name, template["params"], list(cols))
                yield config


def apply_pollutions(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Apply the specified pollutions to the dataset."""
    try:
        polluted_df = df.copy()

        for pollution_params in config.get("pollutions", []):
            polluter = get_polluter(
                polluter_name=pollution_params["name"], **pollution_params["params"]
            )
            logger.info(f"Applying polluter with params: {pollution_params}")
            polluted_df = polluter.apply(polluted_df)

        logger.info("Applying pollutions to dataset...")
        return polluted_df
    except Exception as e:
        logger.error(f"Failed to apply pollutions: {str(e)}")
        sys.exit(1)


def generate_visualizations(f1_df, output_dir):
    logger.info("Generating visualizations...")

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


@click.command()
@click.option(
    "--dataset_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the dataset CSV file",
)
@click.option(
    "--master_config_path",
    type=click.Path(exists=True),
    required=True,
    help="Path to the master configuration YAML file",
)
@click.option(
    "--train-split",
    required=True,
    type=click.Path(exists=True),
    help="Path to the training split CSV file",
)
@click.option(
    "--validation-split",
    required=True,
    type=click.Path(exists=True),
    help="Path to the validation split CSV file",
)
@click.option(
    "--test-split",
    required=True,
    type=click.Path(exists=True),
    help="Path to the test split CSV file",
)
@click.option(
    "--output_dir",
    type=click.Path(),
    required=True,
    help="Directory where configuration files will be saved",
)
@click.option(
    "--samples_per_size", type=int, default=5, help="Number of random samples per combination size"
)
def main(
    dataset_path: str,
    master_config_path: str,
    train_split: str,
    validation_split: str,
    test_split: str,
    output_dir: str,
    samples_per_size: int,
):
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.pkl"

    # Load master configuration and extract templates
    master_config = load_config(master_config_path)
    pollution_templates = get_pollution_templates(master_config)

    # Load all data
    try:
        dataset = pd.read_csv(dataset_path)
        train_split_df = pd.read_csv(train_split)
        validation_split_df = pd.read_csv(validation_split)
        test_split_df = pd.read_csv(test_split)

        # Validate data is not empty
        for df, name in [
            (dataset, "dataset"),
            (train_split_df, "train"),
            (validation_split_df, "validation"),
            (test_split_df, "test"),
        ]:
            if df.empty:
                raise ValueError(f"{name} DataFrame is empty")

    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        sys.exit(1)

    columns = list(dataset.columns)

    # Initialize Model
    matcher = XGBoostMatcher()
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        matcher = XGBoostMatcher.load_model(model_path)
    else:
        logger.info("Training new model...")
        model = matcher.train(dataset, train_split_df, validation_split_df)
        matcher.save_model(model, model_path)

    # Generate configurations and evaluate
    evaluation_results = []
    for name, template in pollution_templates.items():
        for pollution_config in generate_configs(name, template, columns, samples_per_size):
            try:
                polluted_dataset = apply_pollutions(dataset, pollution_config)
                if polluted_dataset.empty:
                    logger.warning(f"Pollution {name} resulted in empty dataset - skipping")
                    continue

                logger.info("Making predictions...")
                predictions = matcher.test(dataset, polluted_dataset, test_split_df)

                logger.info("Calculating metrics...")
                ground_truth = test_split_df["prediction"].values
                f1 = f1_score(ground_truth, predictions)

                result = {
                    "pollution_type": name,
                    "number_of_columns": len(
                        pollution_config["pollutions"][0]["params"]["indices"]
                    ),
                    "f1_score": f1,
                }
                evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Error processing pollution {name}: {str(e)}")
                continue

    if not evaluation_results:
        logger.error("No evaluation results were generated")
        sys.exit(1)

    evaluation_results_df = pd.DataFrame(evaluation_results)
    try:
        generate_visualizations(evaluation_results_df, output_dir)
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")


if __name__ == "__main__":
    main()
