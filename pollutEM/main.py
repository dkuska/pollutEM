import click
from polluters import get_polluter
from utils.config import load_config
from utils.data import load_dataset_and_labels


@click.command()
@click.option("--config-file", default="config.yaml", help="Path to the configuration file.")
def main(config_file):
    """Run the entity matching pipeline with pollution, training, and evaluation."""

    # Step 1: Load Configuration
    config = load_config(config_file)
    click.echo(f"Configuration loaded from {config_file}")

    # Step 2: Load Dataset
    dataset_path = config["dataset"]["dataset_path"]
    label_path = config["dataset"]["label_path"]
    dataset, labels = load_dataset_and_labels(dataset_path, label_path)
    click.echo(f"Dataset loaded from {dataset_path} with columns {dataset.columns}")
    click.echo(f"Labels loaded from {label_path} with columns {labels.columns}")

    type_mapping = dataset.dtypes.to_dict()
    click.echo(f"Column Types of Dataset: {type_mapping}")

    # Step 3: Load Pollutions
    for pollution_params in config.get("pollutions", []):
        polluter = get_polluter(
            polluter_name=pollution_params["name"], **pollution_params["params"]
        )
        click.echo(f"Loaded polluter with params: {pollution_params}")

        dataset = polluter.apply(dataset)


if __name__ == "__main__":
    main()
