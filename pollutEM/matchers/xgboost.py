from typing import Optional, Union
import pickle
from pathlib import Path

from sklearn.metrics import f1_score, roc_curve
import pandas as pd
import numpy as np
import xgboost as xgb


class XGBoostMatcher:
    def __init__(
        self,
        model_params: Optional[dict] = None,
        model: Optional[xgb.Booster] = None,
        numeric_columns: Optional[list[str]] = None,
        optimal_threshold: Optional[float] = None,
    ):
        """
        Initialize XGBoost matcher with optional model parameters

        Args:
            model_params: Dictionary of XGBoost parameters. If None, uses defaults.
        """
        self.model_params = model_params or {}
        self.model = model or None
        self.numeric_columns = numeric_columns or None
        self.optimal_threshold = optimal_threshold or None

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

    def calculate_features(
        self, data1: pd.DataFrame, data2: pd.DataFrame, pairs_df: pd.DataFrame, mode: str = "train"
    ) -> pd.DataFrame:
        """
        Calculate feature differences using the merge approach for data assembly
        but maintaining the original feature calculation style.

        Args:
            data1: First dataset (original/clean data)
            data2: Second dataset (can be same as data1 for training, or polluted for testing)
            pairs_df: DataFrame containing the pairs to compare (with p1, p2 columns)
            mode: Either "train" or "test" - affects how the data is merged

        Returns:
            DataFrame with calculated feature differences
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
            self.numeric_columns = set(self.get_numeric_columns(data1))

        # Create left and right dataframes
        if mode == "train":
            # For training, both merges use the same dataset
            left_features = data1.merge(pairs_df, left_on="id", right_on="p1")
            right_features = data1.merge(pairs_df, left_on="id", right_on="p2")
        else:
            # For testing, merge with respective datasets based on mode
            left_features = data1.merge(pairs_df, left_on="id", right_on="p1")
            right_features = data2.merge(pairs_df, left_on="id", right_on="p2")

        # Prepare the features in the format expected by the original implementation
        combined_features = pd.DataFrame()

        # Add prefixes to distinguish left and right features
        for col in self.numeric_columns:
            if col != "id":  # Skip the ID column
                # Create left_ and right_ prefixed columns
                combined_features[f"left_{col}"] = left_features[col]
                combined_features[f"right_{col}"] = right_features[col]

        # Calculate differences using the original implementation's style
        final_df = pd.DataFrame()
        for col in filter(lambda x: x.startswith("left_"), combined_features.columns):
            base_col = col[5:]  # Remove 'left_' prefix
            final_df[base_col] = combined_features[col] - combined_features[f"right_{base_col}"]

        return final_df

    def train(
        self,
        features: pd.DataFrame,
        train_split: pd.DataFrame,
        validation_split: pd.DataFrame,
        epochs: int = 100,
    ) -> tuple[xgb.Booster, float]:
        # Calculate features using new method
        train_labels = train_split["prediction"].values
        train_features = self.calculate_features(
            features,
            features,
            train_split,
            mode="train",
        )

        valid_labels = validation_split["prediction"].values
        valid_features = self.calculate_features(
            features, features, validation_split, mode="train"
        )

        # Create DMatrix objects
        dtrain = xgb.DMatrix(train_features.values, label=train_labels)
        dvalid = xgb.DMatrix(valid_features.values, label=valid_labels)

        # Train model
        self.model = xgb.train(
            self.model_params, dtrain, epochs, [(dtrain, "train"), (dvalid, "eval")]
        )

        # Find optimal threshold using validation set
        valid_preds = self.model.predict(dvalid)
        _, _, thresholds = roc_curve(valid_labels, valid_preds)
        optimal_idx = np.argmax(
            [f1_score(valid_labels, valid_preds >= thresh) for thresh in thresholds]
        )
        self.optimal_threshold = thresholds[optimal_idx]

        return self.model

    def test(
        self,
        original_features: pd.DataFrame,
        polluted_features: pd.DataFrame,
        test_split: pd.DataFrame,
    ) -> pd.DataFrame:
        test_features = self.calculate_features(
            original_features,
            polluted_features,
            test_split,
            mode="test",
        )
        dtest = xgb.DMatrix(test_features.values)

        ypred = self.model.predict(dtest)

        # Format results
        result = pd.DataFrame({"score": ypred})
        result["prediction"] = 0
        result.loc[result["score"] > self.optimal_threshold, "prediction"] = 1

        return result["prediction"].values

    def save_model(self, model: xgb.Booster, path: Union[str, Path]) -> None:
        """
        Save model and associated state to disk using pickle.

        Args:
            model: Trained XGBoost model
            path: Path where to save the model
        """
        path = Path(path)
        save_dict = {
            "model": model,
            "numeric_columns": self.numeric_columns,
            "model_params": self.model_params,
            "optimal_threshold": self.optimal_threshold,
        }
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> tuple["XGBoostMatcher", xgb.Booster]:
        """
        Load model and associated state from disk using pickle.

        Args:
            path: Path to the saved model

        Returns:
            Tuple of (XGBoostMatchingSolution instance, loaded XGBoost model)
        """
        path = Path(path)
        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        # Create new instance with saved paths
        instance = cls(
            model_params=save_dict["model_params"],
            model=save_dict["model"],
            numeric_columns=save_dict["numeric_columns"],
            optimal_threshold=save_dict["optimal_threshold"],
        )

        return instance
