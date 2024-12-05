# pollutEM

Evaluate Entity Matching Deep Learning Models for polluted numerical data.

This project aims at observing how robust Entity Matching Models are with regards to polluted data, either at training or inference time.

Created as part of the `Table Representation Learning` Seminar at Hasso Plattner Institute Potsdam in the Winter Term 24/25.

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
