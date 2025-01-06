from datetime import datetime
import logging
from pathlib import Path
import sys

import click
from sklearn.metrics import f1_score
import pandas as pd

from utils.config import load_config
from utils.visualization import generate_visualizations
from polluters import apply_pollutions, PollutionConfigGenerator
from matchers import ChatGPTMatcher, XGBoostMatcher


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - MAIN - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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
    # Create base output directory
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_output_path / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.pkl"

    # Load master configuration and extract templates
    master_config = load_config(master_config_path)
    config_generator = PollutionConfigGenerator(master_config)

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

    dataset_columns = list(dataset.columns)

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

    all_configs = list(
        config_generator.get_all_configs(
            all_columns=dataset_columns, samples_per_size=samples_per_size
        )
    )

    for pollution_config in all_configs:
        try:
            name = pollution_config["pollutions"][0]["name"]
            polluted_dataset = apply_pollutions(dataset, pollution_config)
            if polluted_dataset.empty:
                logger.warning(f"Pollution {name} resulted in empty dataset - skipping")
                continue

            predictions = matcher.test(dataset, polluted_dataset, test_split_df)

            ground_truth = test_split_df["prediction"].values
            f1 = f1_score(ground_truth, predictions)

            result = {
                "pollution_type": name,
                "number_of_columns": len(pollution_config["pollutions"][0]["params"]["indices"]),
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
    evaluation_df_path = output_path / "results.csv"
    evaluation_results_df.to_csv(evaluation_df_path, index=False)
    try:
        generate_visualizations(evaluation_results_df, output_path)
    except Exception as e:
        logger.error(f"Error generating visualizations: {str(e)}")


if __name__ == "__main__":
    main()
