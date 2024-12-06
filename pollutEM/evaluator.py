import os
import logging

import click
import pandas as pd
from sklearn.metrics import f1_score

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - EVALUATOR - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_test_features(
    original_data: pd.DataFrame,
    polluted_data: pd.DataFrame,
    test_split: pd.DataFrame,
    mode: str = "mixed",
) -> pd.DataFrame:
    """
    Generate test features by pairing records based on test split.

    Args:
        original_data (pd.DataFrame): Original dataset features with an `id` column.
        polluted_data (pd.DataFrame): Polluted dataset features with an `id` column.
        test_split (pd.DataFrame): Test split containing `p1`, `p2`, and `prediction` columns.
        mode (str): Mode for feature selection. Options:
            - "original": Use both records from the original dataset.
            - "polluted": Use both records from the polluted dataset.
            - "mixed": Use one record from each dataset.

    Returns:
        pd.DataFrame: Combined test features dataframe with the test split `p1`, `p2`, and `prediction` included.
    """
    if mode not in {"original", "polluted", "mixed"}:
        raise ValueError("Invalid mode. Choose from 'original', 'polluted', or 'mixed'.")

    if "id" not in original_data.columns or "id" not in polluted_data.columns:
        raise KeyError("Both `original_data` and `polluted_data` must contain an `id` column.")

    original_data_p1 = original_data.add_suffix("_p1").rename(columns={"id_p1": "p1"})
    original_data_p2 = original_data.add_suffix("_p2").rename(columns={"id_p2": "p2"})
    polluted_data_p1 = polluted_data.add_suffix("_p1").rename(columns={"id_p1": "p1"})
    polluted_data_p2 = polluted_data.add_suffix("_p2").rename(columns={"id_p2": "p2"})

    if mode == "original":
        test_split = test_split.merge(original_data_p1, on="p1").merge(original_data_p2, on="p2")
    elif mode == "polluted":
        test_split = test_split.merge(polluted_data_p1, on="p1").merge(polluted_data_p2, on="p2")
    elif mode == "mixed":
        test_split = test_split.merge(original_data_p1, on="p1").merge(polluted_data_p2, on="p2")
    return test_split


def mock_model_prediction(features):
    """
    Mock function to simulate model predictions.
    For demonstration, we return random predictions.
    Replace this with the actual model logic.
    """
    import numpy as np

    return np.random.randint(0, 2, size=len(features))


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
    "--test_split",
    required=True,
    type=click.Path(exists=True),
    help="Path to the test split CSV file",
)
@click.option(
    "--model", required=True, type=click.Path(), help="Path to the model file (mocked for now)"
)
@click.option(
    "--mode",
    required=True,
    type=click.Choice(["original", "polluted", "mixed"], case_sensitive=False),
    help="Mode for test feature generation",
)
@click.option(
    "--results-dir", type=str, required=True, help="Directory to save evaluation results"
)
def evaluate(original, polluted, test_split, model, mode, results_dir):
    """
    CLI Application to evaluate a model's performance on polluted data.
    """
    logger.info("Loading datasets...")
    original_data = pd.read_csv(original)
    polluted_data = pd.read_csv(polluted)
    test_split_df = pd.read_csv(test_split)

    logger.info("Generating test features...")
    features = generate_test_features(original_data, polluted_data, test_split_df, mode)

    logger.info("Applying model (mocked)...")
    predictions = mock_model_prediction(features)

    logger.info("Calculating F1 Score...")
    ground_truth = test_split_df["prediction"].values
    f1 = f1_score(ground_truth, predictions)

    # Prepare result data
    result = {"original": original, "polluted": polluted, "mode": mode, "f1_score": f1}

    # Ensure the results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # Store results in a CSV
    results_file = os.path.join(results_dir, "evaluation_results.csv")
    if not os.path.exists(results_file):
        # If file doesn't exist, write headers
        results_df = pd.DataFrame([result])
        results_df.to_csv(results_file, mode="w", index=False)
    else:
        # If file exists, append results
        results_df = pd.DataFrame([result])
        results_df.to_csv(results_file, mode="a", header=False, index=False)

    logger.info(f"Evaluation complete. F1 Score: {f1}")


if __name__ == "__main__":
    evaluate()
