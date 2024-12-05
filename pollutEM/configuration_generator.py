import click
import yaml
import pandas as pd
import random
from itertools import combinations
from pathlib import Path
from typing import List, Dict, Any


def load_master_config(config_path: str) -> Dict[str, Any]:
    """Load the master configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_dataset_columns(dataset_path: str) -> List[str]:
    """Get column names from the dataset."""
    df = pd.read_csv(dataset_path)
    return list(df.columns)


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


def generate_configs(
    name: str, template: Dict, all_columns: List[str], output_dir: Path, samples_per_size: int = 5
):
    """Generate random samples of column combinations for each size."""
    # Get applicable columns that exist in the dataset
    applicable_columns = [col for col in template["applicable_columns"] if col in all_columns]

    if not applicable_columns:
        click.echo(f"Warning: No applicable columns found for {name}")
        return

    click.echo(f"  Applicable columns: {', '.join(applicable_columns)}")

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
                filename = f"{name}_{'_'.join(cols)}.yaml"
                with open(output_dir / filename, "w") as f:
                    yaml.dump(config, f, default_flow_style=False)


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
    "--output_dir",
    type=click.Path(),
    required=True,
    help="Directory where configuration files will be saved",
)
@click.option(
    "--samples_per_size", type=int, default=5, help="Number of random samples per combination size"
)
def main(dataset_path: str, master_config_path: str, output_dir: str, samples_per_size: int):
    """Generate pollution configurations based on master config templates."""
    # Load master configuration and extract templates
    master_config = load_master_config(master_config_path)
    pollution_templates = get_pollution_templates(master_config)

    # Get dataset columns
    columns = get_dataset_columns(dataset_path)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate configurations for each pollution type
    for name, template in pollution_templates.items():
        click.echo(f"Generating configurations for {name}")
        param_str = ", ".join(f"{k}={v}" for k, v in template["params"].items() if k != "level")
        click.echo(f"  With parameters: {param_str}")
        generate_configs(name, template, columns, output_path, samples_per_size)


if __name__ == "__main__":
    main()
