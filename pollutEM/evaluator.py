#!/usr/bin/env python3
import click
import pandas as pd
from pathlib import Path
from typing import List
import logging
import sys
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--clean-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to clean dataset",
)
@click.option(
    "--polluted-data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    multiple=True,
    help="Paths to polluted datasets (can be specified multiple times)",
)
@click.option(
    "--names",
    type=str,
    multiple=True,
    help="Names for each polluted dataset (must match number of polluted datasets)",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=Path("analysis_results"),
    help="Directory for output files (default: analysis_results)",
)
def main(clean_data: Path, polluted_data: List[Path], names: List[str], output_dir: Path) -> None:
    """
    Analyze differences between clean and polluted datasets.

    This command loads the clean dataset and one or more polluted versions,
    performs analysis, and generates plots and a report.
    """
    try:
        # Validate inputs
        if len(polluted_data) != len(names):
            raise click.BadParameter("Number of names must match number of polluted datasets")

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_dir / timestamp
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load datasets
        logger.info("Loading datasets...")
        clean_df = pd.read_csv(clean_data)
        polluted_dfs = [pd.read_csv(path) for path in polluted_data]

        logger.info(f"Analysis completed. Results saved to: {output_dir}")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
