import logging
from pathlib import Path

import click
import pandas as pd
from sklearn.metrics import f1_score

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
def train_and_evaluate(original, polluted, train_split, test_split, mode, results_dir):
    """Train a new XGBoost matcher model and evaluate its performance."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Training phase
    logger.info("Loading training data...")
    features_df = pd.read_csv(original)
    train_split_df = pd.read_csv(train_split)

    logger.info("Initializing and training XGBoost matcher...")
    matcher = XGBoostMatcher()
    matcher.train(features_df, train_split_df)

    # Evaluation phase
    logger.info("Loading test datasets...")
    original_data = pd.read_csv(original)
    polluted_data = pd.read_csv(polluted)
    test_split_df = pd.read_csv(test_split)

    logger.info("Making predictions...")
    predictions = matcher.predict(original_data, polluted_data, test_split_df, mode)
    binary_predictions = (predictions >= 0.5).astype(int)

    logger.info("Calculating metrics...")
    ground_truth = test_split_df["prediction"].values
    f1 = f1_score(ground_truth, binary_predictions)

    # Store results
    result = {
        "original": original,
        "polluted": polluted,
        "mode": mode,
        "f1_score": f1,
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
