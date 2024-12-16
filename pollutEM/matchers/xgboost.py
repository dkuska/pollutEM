import pickle
from pathlib import Path
from typing import Union, Optional

import pandas as pd
import numpy as np
import xgboost as xgb

from .base_matcher import BaseMatcher


class XGBoostMatcher(BaseMatcher):
    def __init__(self, model_params: Optional[dict] = None):
        """
        Initialize XGBoost matcher with optional model parameters

        Args:
            model_params: Dictionary of XGBoost parameters. If None, uses defaults.
        """
        self.model_params = model_params or {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": 6,
            "learning_rate": 0.1,
            "n_estimators": 100,
        }
        self.model = xgb.XGBClassifier(**self.model_params)

        # TODO: Extend this to work with other datasets!
        self.feature_columns = [
            "depth",
            "depth_uncertainty",
            "horizontal_uncertainty",
            "used_phase_count",
            "used_station_count",
            "standard_error",
            "azimuthal_gap",
            "minimum_distance",
            "mag_value",
        ]

    def _create_train_features(
        self, features_df: pd.DataFrame, pairs_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create feature differences for training pairs

        Args:
            features_df: DataFrame containing feature data
            pairs_df: DataFrame containing pairs to compare

        Returns:
            DataFrame with feature differences
        """
        # Convert IDs to int
        features_df = features_df.copy()
        pairs_df = pairs_df.copy()
        features_df["id"] = features_df["id"].astype(int)
        pairs_df["p1"] = pairs_df["p1"].astype(int)
        pairs_df["p2"] = pairs_df["p2"].astype(int)

        # Merge features for both elements in pairs
        p1_features = features_df.merge(pairs_df, left_on="id", right_on="p1")
        p2_features = features_df.merge(pairs_df, left_on="id", right_on="p2")

        # Calculate differences
        feature_differences = {}
        for col in self.feature_columns:
            feature_differences[f"{col}_diff"] = p1_features[col] - p2_features[col]

        return pd.DataFrame(feature_differences)

    def _create_test_features(
        self,
        original_data: pd.DataFrame,
        polluted_data: pd.DataFrame,
        test_split: pd.DataFrame,
        mode: str = "mixed",
    ) -> pd.DataFrame:
        """
        Generate test features by pairing records based on test split.

        Args:
            original_data: Original dataset features with an `id` column
            polluted_data: Polluted dataset features with an `id` column
            test_split: Test split containing `p1`, `p2`, and `prediction` columns
            mode: Mode for feature selection

        Returns:
            DataFrame with combined test features
        """
        if mode not in {"original", "polluted", "mixed"}:
            raise ValueError("Invalid mode. Choose from 'original', 'polluted', or 'mixed'.")

        if "id" not in original_data.columns or "id" not in polluted_data.columns:
            raise KeyError("Both `original_data` and `polluted_data` must contain an `id` column.")

        original_data_p1 = original_data.add_suffix("_p1").rename(columns={"id_p1": "p1"})
        original_data_p2 = original_data.add_suffix("_p2").rename(columns={"id_p2": "p2"})
        polluted_data_p1 = polluted_data.add_suffix("_p1").rename(columns={"id_p1": "p1"})
        polluted_data_p2 = polluted_data.add_suffix("_p2").rename(columns={"id_p2": "p2"})

        if mode == "original":
            features = test_split.merge(original_data_p1, on="p1").merge(original_data_p2, on="p2")
        elif mode == "polluted":
            features = test_split.merge(polluted_data_p1, on="p1").merge(polluted_data_p2, on="p2")
        else:  # mixed mode
            features = test_split.merge(original_data_p1, on="p1").merge(polluted_data_p2, on="p2")

        # Calculate differences for the relevant columns
        feature_differences = {}
        for col in self.feature_columns:
            feature_differences[f"{col}_diff"] = features[f"{col}_p1"] - features[f"{col}_p2"]

        return pd.DataFrame(feature_differences)

    def train(self, features_df: pd.DataFrame, train_split_df: pd.DataFrame) -> None:
        """
        Train the XGBoost model

        Args:
            features_df: DataFrame containing feature data
            train_split_df: DataFrame containing training pairs and labels
        """
        X = self._create_train_features(features_df, train_split_df)
        y = train_split_df["prediction"]
        self.model.fit(X, y)

    def predict(
        self,
        original_data: pd.DataFrame,
        polluted_data: pd.DataFrame,
        pairs_df: pd.DataFrame,
        mode: str = "mixed",
    ) -> np.ndarray:
        """
        Make predictions for given pairs

        Args:
            original_data: Original dataset features
            polluted_data: Polluted dataset features
            pairs_df: DataFrame containing pairs to predict
            mode: Mode for feature selection

        Returns:
            Array of predictions
        """
        X = self._create_test_features(original_data, polluted_data, pairs_df, mode)
        return self.model.predict_proba(X)[:, 1]

    def predict_proba(
        self,
        original_data: pd.DataFrame,
        polluted_data: pd.DataFrame,
        pairs_df: pd.DataFrame,
        mode: str = "mixed",
    ) -> np.ndarray:
        """
        Get probability predictions for given pairs

        Args:
            original_data: Original dataset features
            polluted_data: Polluted dataset features
            pairs_df: DataFrame containing pairs to predict
            mode: Mode for feature selection

        Returns:
            Array of probability predictions
        """
        X = self._create_test_features(original_data, polluted_data, pairs_df, mode)
        return self.model.predict_proba(X)

    def save_model(self, path: Union[str, Path]) -> None:
        """
        Save model to disk using pickle

        Args:
            path: Path where to save the model
        """
        path = Path(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load_model(cls, path: Union[str, Path]) -> "XGBoostMatcher":
        """
        Load model from disk using pickle

        Args:
            path: Path to the saved model

        Returns:
            Loaded XGBoostMatcher instance
        """
        path = Path(path)
        with open(path, "rb") as f:
            return pickle.load(f)
