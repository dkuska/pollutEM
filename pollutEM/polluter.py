#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path
from typing import Dict, Any
import yaml
import sys
import logging

from polluters import get_polluter


# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - POLLUTER - %(levelname)s - %(message)s"
)
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


@click.command()
@click.option(
    "--input-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to input CSV file",
)
@click.option(
    "--config-file",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to configuration YAML file",
)
@click.option(
    "--output-file",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path for output CSV file",
)
def main(input_file: Path, config_file: Path, output_file: Path) -> None:
    """
    Apply specified pollutions to a dataset.

    This tool takes a clean dataset and applies various pollutions as specified
    in the configuration file, then saves the polluted dataset to the output path.
    """
    try:
        logger.info(f"Loading input file: {input_file}")
        df = pd.read_csv(input_file)

        logger.info(f"Loading configuration from: {config_file}")
        config = load_config(config_file)

        polluted_df = apply_pollutions(df, config)

        logger.info(f"Saving polluted dataset to: {output_file}")
        polluted_df.to_csv(output_file, index=False)

        logger.info("Process completed successfully")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
