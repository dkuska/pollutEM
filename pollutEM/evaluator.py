import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import f1_score
import xgboost as xgb

from matchers.xgboost import XGBoostMatcher

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - EVALUATOR - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--original",
    required=True,
    type=click.Path(exists=True),
    help="Path to the original dataset CSV file",
)
@click.option(
    "--polluted",
    required=True,
    type=click.Path(exists=True),
    help="Path to the polluted dataset CSV file",
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
    "--mode",
    required=True,
    type=click.Choice(["original", "polluted", "mixed"], case_sensitive=False),
    help="Mode for test feature generation",
)
@click.option(
    "--results-dir",
    type=str,
    required=True,
    help="Directory to save evaluation results and model",
)
def train_and_evaluate(
    original: str,
    polluted: str,
    train_split: str,
    validation_split: str,
    test_split: str,
    mode: str,
    results_dir: str,
) -> None:
    """
    Train a new XGBoost matcher model (or load existing) and evaluate its performance.

    Args:
        original: Path to original dataset
        polluted: Path to polluted dataset
        train_split: Path to training split
        validation_split: Path to validation split
        test_split: Path to test split
        mode: Mode for feature selection
        results_dir: Directory to save results
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_path = results_dir / "model.pkl"

    matcher = XGBoostMatcher()

    # Training phase (or loading existing model)
    if model_path.exists():
        logger.info(f"Loading existing model from {model_path}")
        matcher = XGBoostMatcher.load_model(model_path)
    else:
        logger.info("Training new model...")
        logger.info("Loading training data...")
        features_df = pd.read_csv(original)
        train_split_df = pd.read_csv(train_split)
        validation_split_df = pd.read_csv(validation_split)

        logger.info("Initializing and training XGBoost matcher...")
        model, optimal_threshold = matcher.train(features_df, train_split_df, validation_split_df)
        logger.info(f"Optimal Threshold: {optimal_threshold}")

        # Save the trained model
        logger.info(f"Saving model to {model_path}")
        matcher.save_model(model, model_path)

    # Evaluation phase
    logger.info("Loading test datasets...")
    original_data = pd.read_csv(original)
    polluted_data = pd.read_csv(polluted)
    test_split_df = pd.read_csv(test_split)

    logger.info("Making predictions...")
    predictions = matcher.test(original_data, polluted_data, test_split_df)

    logger.info("Calculating metrics...")
    ground_truth = test_split_df["prediction"].values
    f1 = f1_score(ground_truth, predictions)

    # Store results
    result = {
        "original": original,
        "polluted": polluted,
        "mode": mode,
        "f1_score": f1,
        "model_path": str(model_path),  # Add model path to results
    }

    results_file = results_dir / "evaluation_results.csv"
    results_df = pd.DataFrame([result])

    if not results_file.exists():
        results_df.to_csv(results_file, index=False)
    else:
        results_df.to_csv(results_file, mode="a", header=False, index=False)

    logger.info(f"Training and evaluation complete. F1 Score: {f1}")
    logger.info(f"Results saved to {results_dir}")


if __name__ == "__main__":
    train_and_evaluate()
