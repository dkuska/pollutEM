from dataclasses import dataclass

import pandas as pd

from .ditto_light.dataset import DittoDataset
from .ditto_light.ditto import train as ditto_train


@dataclass
class DittoHyperParameters:
    batch_size: int = 64
    lm: str = "distilbert"
    alpha_aug: float = 0.8
    lr: float = 3e-5
    n_epochs: int = 20
    logdir: str = "/logs"
    task: str = "blablabla"
    fp16: bool = False


class DittoMatcher:
    def __init__(self):
        self.task = "blablabla"
        self.run_id = 1234
        self.batch_size = 64
        self.max_len = 256
        self.lr = 3e-5
        self.n_epochs = 20
        self.finetuning = True
        self.save_model = True
        self.log_dir = "models"
        self.language_model = "distilbert"
        self.fp16 = True
        self.da = None
        self.alpha_aug = 0.8
        self.dk = None
        self.summarize = True
        self.size = None

        self.hyperparams = DittoHyperParameters(
            batch_size=self.batch_size,
            lm=self.language_model,
            alpha_aug=self.alpha_aug,
            lr=self.lr,
            n_epochs=self.n_epochs,
            logdir=self.log_dir,
            task=self.task,
        )

        self.numeric_columns = None

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> list[str]:
        """Get numeric columns following original implementation."""
        numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
        cols = list(df.select_dtypes(include=numerics).columns)
        if "id" not in cols:
            cols.append("id")

        # Return None if only ID column exists
        if len(cols) == 1 and cols[0] == "id":
            return None

        return cols

    def format_record(self, df_row: pd.Series, columns_to_format: set[str]):
        """
        Format a single DataFrame row into the required string format.

        Args:
            df_row (pandas.Series): A single row from a DataFrame
            columns_to_format (set[str]): Set of column names to include in formatting

        Returns:
            str: A formatted string of the record
        """
        formatted_parts = []
        for col in columns_to_format:
            if col != "id":
                value = str(df_row[col])
                formatted_parts.append(f'COL {col} VAL "{value}"')
        return " ".join(formatted_parts)

    def calculate_features(
        self, data1: pd.DataFrame, data2: pd.DataFrame, pairs_df: pd.DataFrame, mode: str = "train"
    ) -> list:
        """
        Calculate feature differences and format them as strings using the pair-based approach.
        Args:
            data1: First dataset (original/clean data)
            data2: Second dataset (can be same as data1 for training, or polluted for testing)
            pairs_df: DataFrame containing the pairs to compare (with p1, p2, prediction columns)
            mode: Either "train" or "test" - affects how the data is merged
        Returns:
            List of strings containing formatted records with their labels
        """
        # Input validation
        if "id" not in data1.columns or "id" not in data2.columns:
            raise KeyError("Both datasets must contain an 'id' column")

        # Ensure IDs are integers
        data1 = data1.copy()
        data2 = data2.copy()
        pairs_df = pairs_df.copy()
        data1["id"] = data1["id"].astype(int)
        data2["id"] = data2["id"].astype(int)
        pairs_df["p1"] = pairs_df["p1"].astype(int)
        pairs_df["p2"] = pairs_df["p2"].astype(int)

        # Get numeric columns if not already set
        if self.numeric_columns is None:
            self.numeric_columns = set(DittoMatcher.get_numeric_columns(data1))

        # Pre-format all records from data1 and data2 into dictionaries
        data1_formatted = {
            row["id"]: self.format_record(row, self.numeric_columns) for _, row in data1.iterrows()
        }

        # For training, use data1 for both dictionaries
        if mode == "train":
            data2_formatted = data1_formatted
        # For testing, format data2 separately
        else:
            data2_formatted = {
                row["id"]: self.format_record(row, self.numeric_columns)
                for _, row in data2.iterrows()
            }

        # Create formatted strings directly from pairs_df
        formatted_records = []
        for _, row in pairs_df.iterrows():
            p1_id = row["p1"]
            p2_id = row["p2"]
            label = row["prediction"]

            # Get pre-formatted strings from dictionaries
            left_formatted = data1_formatted[p1_id]
            right_formatted = data2_formatted[p2_id]

            # Combine with tab separators and label
            combined_record = f"{left_formatted}\t{right_formatted}\t{label}"
            formatted_records.append(combined_record)

        return formatted_records

    def train(
        self,
        features: pd.DataFrame,
        train_split: pd.DataFrame,
        validation_split: pd.DataFrame,
        test_split: pd.DataFrame,
    ):

        train_set = DittoDataset(
            self.calculate_features(
                features,
                features,
                train_split,
                mode="train",
            ),
            max_len=self.max_len,
            size=self.size,
            lm=self.language_model,
            da=self.da,
        )

        validation_set = DittoDataset(
            self.calculate_features(
                features,
                features,
                validation_split,
                mode="train",
            ),
            max_len=self.max_len,
            size=self.size,
            lm=self.language_model,
            da=self.da,
        )

        test_set = DittoDataset(
            self.calculate_features(
                features,
                features,
                test_split,
                mode="train",
            ),
            max_len=self.max_len,
            size=self.size,
            lm=self.language_model,
            da=self.da,
        )

        ditto_train(
            trainset=train_set,
            validset=validation_set,
            testset=test_set,
            run_tag=None,
            hp=self.hyperparams,
        )
