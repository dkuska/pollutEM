import pandas as pd


def load_dataset_and_labels(dataset_path, label_path):
    return pd.read_csv(dataset_path), pd.read_csv(label_path)
