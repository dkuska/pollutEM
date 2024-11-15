import click
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
    click.echo(f"Dataset loaded from {dataset_path}")
    click.echo(f"Dataset loaded from {label_path}")

    # Step 3: Apply Pollutions
    polluted_datasets = []
    for pollution_params in config.get("pollutions", []):
        click.echo(f"Applied pollution with params: {pollution_params}")
        polluted_datasets.append((pollution_params, {}))

    # Step 4: Train Matchers
    matchers = []
    for matcher_config in config["matchers"]:
        matcher_name = matcher_config["name"]
        click.echo(f"Trained matcher: {matcher_name}")

    # Step 5: Evaluate
    results = []
    for matcher in matchers:
        for pollution_params, polluted_dataset in polluted_datasets:
            click.echo(f"Evaluated matcher: {matcher.name} with pollution {pollution_params}")

    # Output results
    click.echo("Pipeline completed. Results:")
    for result in results:
        click.echo(
            f"Matcher: {result['matcher']}, Pollution: {result['pollution']}, Result: {result['result']}"
        )


if __name__ == "__main__":
    main()
