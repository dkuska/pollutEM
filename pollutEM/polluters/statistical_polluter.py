from typing import Optional
import numpy as np
import pandas as pd

from .base_polluter import BasePolluter


class ZScoreNormalizationPolluter(BasePolluter):
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std
        super().__init__(self.apply_zscore)

    def apply_zscore(self, df: pd.DataFrame, column: str):
        if self.mean is None:
            self.mean = df[column].mean()
        if self.std is None:
            self.std = df[column].std()

        return (df[column] - self.mean) / (self.std if self.std != 0 else 1)

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
            np.int32: np.float32,
            np.float32: np.float32,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["column"]


class MinMaxNormalizationPolluter(BasePolluter):
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(self.apply_minmax)

    def apply_minmax(self, df: pd.DataFrame, column: str):
        if self.min_val is None:
            self.min_val = df[column].min()
        if self.max_val is None:
            self.max_val = df[column].max()

        denominator = self.max_val - self.min_val
        if denominator == 0:
            return np.zeros(len(df))
        return (df[column] - self.min_val) / denominator

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["column"]
