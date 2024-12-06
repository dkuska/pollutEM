import click
import pandas as pd
from sklearn.metrics import f1_score


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
def evaluate(original, polluted, test_split, model, mode):
    """
    CLI Application to evaluate a model's performance on polluted data.
    """
    click.echo("Loading datasets...")
    original_data = pd.read_csv(original)
    polluted_data = pd.read_csv(polluted)
    test_split_df = pd.read_csv(test_split)

    click.echo("Generating test features...")
    features = generate_test_features(original_data, polluted_data, test_split_df, mode)

    click.echo("Applying model (mocked)...")
    predictions = mock_model_prediction(features)

    click.echo("Calculating F1 Score...")
    ground_truth = test_split_df["prediction"].values
    f1 = f1_score(ground_truth, predictions)
    click.echo(f"F1 Score: {f1}")


if __name__ == "__main__":
    evaluate()
