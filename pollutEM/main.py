from datetime import datetime
import logging
from pathlib import Path
import sys
import os

import click
from dotenv import load_dotenv
from sklearn.metrics import f1_score, recall_score, precision_score
import pandas as pd
from tqdm import tqdm

from utils.config import load_config
from utils.random import set_seed
from utils.visualization import generate_visualizations
from polluters import apply_pollutions, PollutionConfigGenerator
from matchers import ChatGPTMatcher, XGBoostMatcher


load_dotenv()

# Set up logging
logging.getLogger("openai").disabled = True
logging.getLogger("httpx").disabled = True
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
@click.option("--seed", type=int, default=42, help="Seed to use for RNG")
def main(
    dataset_path: str,
    master_config_path: str,
    train_split: str,
    validation_split: str,
    test_split: str,
    output_dir: str,
    samples_per_size: int,
    seed: int,
):
    # Set seed for reproducability
    set_seed(seed)

    # Create base output directory
    base_output_path = Path(output_dir)
    base_output_path.mkdir(parents=True, exist_ok=True)

    # Create sub-directory where results of this run are saved
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = base_output_path / f"run_{timestamp}"
    output_path.mkdir(parents=True, exist_ok=True)
    model_path = output_path / "model.pkl"
    logger.info(f"Saving run results to {output_path}")

    # Load master configuration and extract templates
    master_config = load_config(master_config_path)
    config_generator = PollutionConfigGenerator(master_config)

    # Load and validate all data
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

    # Initialize Models and Matchers
    # TODO: Make Matchers configurable!
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        xgboost_matcher = XGBoostMatcher.load_model(model_path)
    else:
        logger.info("Training new model...")
        xgboost_matcher = XGBoostMatcher()
        model = xgboost_matcher.train(dataset, train_split_df, validation_split_df)
        xgboost_matcher.save_model(model, model_path)

    chatgpt_matcher = ChatGPTMatcher(api_key=os.environ.get("API_KEY"))
    matchers = [xgboost_matcher]  # , chatgpt_matcher]

    # Generate configurations
    dataset_columns = list(dataset.columns)
    all_configs = list(
        config_generator.get_all_configs(
            all_columns=dataset_columns, samples_per_size=samples_per_size
        )
    )

    # Evaluate all Matchers for all Configurations
    evaluation_results = []
    for matcher in matchers:
        # Generate Baseline without Data Pollution
        predictions = matcher.test(dataset, dataset, test_split_df)
        ground_truth = test_split_df["prediction"].values
        f1 = f1_score(ground_truth, predictions)
        precision = precision_score(ground_truth, predictions)
        recall = recall_score(ground_truth, predictions)

        evaluation_results.append(
            {
                "matcher": matcher.name,
                "pollution_type": "None",
                "number_of_columns": 0,
                "params": "",
                "f1_score": f1,
                "precision": precision,
                "recall": recall,
            }
        )

        for pollution_config in tqdm(
            all_configs, desc=f"Processing Configuration with Matcher {matcher.name}"
        ):
            try:
                pollution_name = pollution_config["pollutions"][0]["name"]
                number_of_columns = len(pollution_config["pollutions"][0]["params"]["indices"])
                keys_to_exclude = ["indices", "level"]
                pollution_params = {
                    k: v
                    for k, v in pollution_config["pollutions"][0]["params"].items()
                    if k not in keys_to_exclude
                }
                pollution_param_string = str(pollution_params)

                polluted_dataset = apply_pollutions(dataset, pollution_config)
                if polluted_dataset.empty:
                    logger.warning(
                        f"Pollution {pollution_name} resulted in empty dataset - skipping"
                    )
                    continue

                predictions = matcher.test(dataset, polluted_dataset, test_split_df)

                ground_truth = test_split_df["prediction"].values
                f1 = f1_score(ground_truth, predictions)
                precision = precision_score(ground_truth, predictions)
                recall = recall_score(ground_truth, predictions)

                result = {
                    "matcher": matcher.name,
                    "pollution_type": pollution_name,
                    "number_of_columns": number_of_columns,
                    "params": pollution_param_string,
                    "f1_score": f1,
                    "precision": precision,
                    "recall": recall,
                }
                evaluation_results.append(result)
            except Exception as e:
                logger.error(f"Error processing pollution {pollution_name}: {str(e)}")
                continue

    if not evaluation_results:
        logger.error("No evaluation results were generated")
        sys.exit(1)

    # Save evaluation data and generate Visualization
    metrics_df = pd.DataFrame(evaluation_results)
    evaluation_csv_path = output_path / "results.csv"
    metrics_df.to_csv(evaluation_csv_path, index=False)
    try:
        generate_visualizations(metrics_df, output_dir=output_path, metric="precision")
        generate_visualizations(metrics_df, output_dir=output_path, metric="recall")
        generate_visualizations(metrics_df, output_dir=output_path, metric="f1_score")
    except Exception as e:
        logger.error(f"Error generating visualizations: {e}")


if __name__ == "__main__":
    main()
