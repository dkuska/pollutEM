# pollutEM

Evaluate Entity Matching Deep Learning Models for polluted numerical data.

This project aims at observing how robust Entity Matching Models are with regards to polluted data, either at training or inference time.

Created as part of the `Table Representation Learning` Seminar at Hasso Plattner Institute Potsdam in the Winter Term 24/25.

## Entity Matching Pipeline

The pipeline.py script orchestrates a complete workflow for evaluating an entity matching model's robustness to polluted data. It generates multiple configurations for data pollution, applies these configurations to a dataset, evaluates the model's performance on the polluted datasets, and generates visualizations and reports.
Usage

```bash
python pipeline.py \
    --dataset-path /path/to/original/dataset.csv \
    --master-config-path /path/to/master_config.yaml \
    --config-dir /path/to/output/configs \
    --samples-per-size 5 \
    --model-path /path/to/trained/model.pkl \
    --test-split-path /path/to/test_split.csv \
    --results-dir /path/to/output/results \
    --mode mixed
```

### Parameters

`--dataset-path`: Path to the original (clean) dataset in CSV format.

`--master-config-path`: Path to the master configuration YAML file containing the pollution definitions.

`--config-dir`: Directory where individual pollution configurations will be saved after random sampling from the master configuration.

`--samples-per-size`: Number of random combinations of columns to generate for each pollution configuration size (default: 5).

`--model-path`: Path to the trained model that will be evaluated.

`--test-split-path`: Path to the test split CSV file. It should contain record pair IDs (p1, p2) along with the ground truth prediction column.

`--results-dir`: Directory where evaluation results (F1 scores) and reports will be stored.

`--mode`: Mode of feature generation for the evaluation. It can be one of:
    original: Use the original dataset.
    polluted: Use the polluted dataset.
    mixed: Use a combination of both the original and polluted datasets.

### Workflow Overview

Generate Pollution Configurations: The script starts by generating individual pollution configurations from a master configuration file. These configurations define how different pollutions (e.g., noise, rounding) should be applied to various columns in the dataset. Random combinations of applicable columns are generated and saved as individual configuration files.

Apply Pollution to Dataset: For each generated configuration, the script applies the defined pollution to the original dataset. This step generates multiple polluted versions of the dataset based on the different pollution configurations.

Evaluate Model: The model is then evaluated on each polluted dataset. The script uses the provided test split (`test-split-path`) to generate feature vectors from the polluted data and applies the trained model. The F1 score of the model’s predictions is calculated and stored in the results directory.

Generate Reports and Visualizations: After all evaluations are complete, the script generates F1 score reports and creates visualizations to summarize the model’s performance across different pollution configurations and modes.

## Configuration Generator

The `configuration_generator.py` script helps create individual pollution configurations from a master configuration file. It uses random sampling to create a manageable number of configurations for different column combinations.

### Usage

```bash
python configuration_generator.py \
    --dataset_path /path/to/your/dataset.csv \
    --master_config_path /path/to/master_config.yaml \
    --output_dir /path/to/output/directory \
    --samples_per_size 5
```

### Parameters

- `dataset_path`: Path to your CSV dataset
- `master_config_path`: Path to the master configuration file
- `output_dir`: Directory where individual configuration files will be saved
- `samples_per_size`: Number of random combinations to generate for each combination size (default: 5)

### Master Configuration Format

The master configuration file defines which pollutions should be applied and to which columns. Example:

```yaml
pollutions:
  - name: "GaussianNoise"
    params:
      level: "column"
      mean: 1.0
      std_dev: 0.05
    applicable_columns:
      - "depth"
      - "depth_uncertainty"

  - name: "Rounding"
    params:
      level: "column"
      decimal_places: 1
    applicable_columns:
      - "depth"
      - "mag_value"
```

Each pollution type requires:

- `name`: Name of the pollution to apply
- `params`: Parameters specific to the pollution type
- `applicable_columns`: List of columns this pollution can be applied to

The script will generate individual configuration files for random combinations of the applicable columns, creating separate files for each pollution type and column combination.

### Output

The script creates individual YAML files in the specified output directory. Each file contains a single pollution configuration. Files are named according to the pattern `{pollution_type}_{column1}_{column2}.yaml`.

Example output file:

```yaml
pollutions:
  - name: "GaussianNoise"
    params:
      level: "column"
      mean: 1.0
      std_dev: 0.05
      indices:
        - "depth"
        - "depth_uncertainty"
```

## Data Polluter

The `polluter.py` script applies data pollution configurations to your dataset. It reads a clean dataset and a configuration file, applies the specified pollutions, and outputs the polluted dataset.

### Usage

```bash
python polluter.py \
    --input-file /path/to/clean/dataset.csv \
    --config-file /path/to/pollution/config.yaml \
    --output-file /path/to/output/polluted_dataset.csv
```

### Parameters

- `--input-file`: Path to the clean input CSV dataset
- `--config-file`: Path to the YAML configuration file specifying the pollutions to apply
- `--output-file`: Path where the polluted dataset will be saved

### Configuration Format

The configuration file should specify one or more pollutions to apply to the dataset. Example:

```yaml
pollutions:
  - name: "GaussianNoise"
    params:
      level: "column"
      indices: ["depth"]
      mean: 1.0
      std_dev: 0.05
```

Each pollution requires:

- `name`: The type of pollution to apply
- `params`: Parameters specific to that pollution type, including:
  - `level`: The level at which to apply the pollution (e.g., "column")
  - `indices`: List of columns to pollute
  - Additional parameters specific to the pollution type

### Example Workflow

1. Start with a clean dataset:

```bash
features.csv
```

2. Apply pollution using a configuration:

```bash
python polluter.py \
    --input-file features.csv \
    --config-file configs/gaussian_depth.yaml \
    --output-file polluted_features.csv
```

3. The script will:
   - Load your dataset
   - Apply the specified pollutions
   - Save the polluted dataset
   - Provide logging information about the process

## Evaluator

The `evaluator.py` script evaluates a model's performance on polluted data. It takes an original dataset, a polluted dataset, and a test split, then generates the corresponding feature vectors based on the specified mode. After applying the model (mocked for now), it calculates and displays the F1 score against the ground-truth data.

### Usage

```bash
python evaluator.py \
    --original /path/to/original_dataset.csv \
    --polluted /path/to/polluted_dataset.csv \
    --test_split /path/to/test_split.csv \
    --model /path/to/model.pkl \
    --mode original
```

### Parameters

- `--original`: Path to the original, clean CSV dataset
- `--polluted`: Path to the polluted CSV dataset
- `--test_split`: Path to the CSV file containing the test split with columns `p1`, `p2`, and `prediction`
- `--model`: Path to the model file (mocked for now, but should be replaced with actual model logic)
- `--mode`: Mode to determine how the test features are generated (`original`, `polluted`, or `mixed`)

### Mode Descriptions

- `original`: Generates test features from the original dataset
- `polluted`: Generates test features from the polluted dataset
- `mixed`: Generates test features using one record from the original and one from the polluted dataset

### Example Workflow

1. Start with your clean dataset and a polluted version of it:

```bash
original_dataset.csv
polluted_dataset.csv
```

2. Also, have your test split, which contains record pairs and the corresponding ground-truth prediction:

```bash
test_split.csv
```

3. Run the evaluator:

```bash
python evaluator.py \
    --original original_dataset.csv \
    --polluted polluted_dataset.csv \
    --test_split test_split.csv \
    --model model.pkl \
    --mode mixed
```

4. The script will:
   - Load the datasets and test split
   - Generate feature vectors based on the specified mode
   - Apply the model to generate predictions (mocked for now)
   - Calculate the F1 score of the predictions against the ground-truth values and display the result
