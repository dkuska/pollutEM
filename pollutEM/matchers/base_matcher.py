from abc import ABC, abstractmethod
from typing import Union
from pathlib import Path

import numpy as np
import pandas as pd


class BaseMatcher(ABC):
    """Abstract base class for all matcher models"""

    @abstractmethod
    def train(self, features_df: pd.DataFrame, train_split_df: pd.DataFrame) -> None:
        """Train the model with given features and training pairs"""
        pass

    @abstractmethod
    def predict(self, features_df: pd.DataFrame, pairs_df: pd.DataFrame) -> np.ndarray:
        """Make predictions for given pairs"""
        pass

    @abstractmethod
    def save_model(self, path: Union[str, Path]) -> None:
        """Save the model to disk"""
        pass

    @abstractmethod
    def load_model(self, path: Union[str, Path]) -> None:
        """Load the model from disk"""
        pass

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
        raise NotImplementedError

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
            mode: Mode for feature selection. Options:
                - "original": Use both records from the original dataset
                - "polluted": Use both records from the polluted dataset
                - "mixed": Use one record from each dataset

        Returns:
            DataFrame with combined test features
        """
        raise NotImplementedError
