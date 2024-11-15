from typing import Type, Optional
import numpy as np
import polars as pl

from .base_polluter import BasePolluter


class ZScoreNormalizationPolluter(BasePolluter):
    def __init__(self, mean: Optional[float] = None, std: Optional[float] = None):
        self.mean = mean
        self.std = std
        super().__init__(self.apply_zscore)

    def apply_zscore(self, df: pl.DataFrame, column: str):
        if self.mean is None:
            self.mean = df[column].mean()
        if self.std is None:
            self.std = df[column].std()

        return (df[column] - self.mean) / (self.std if self.std != 0 else 1)

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Float64}


class MinMaxNormalizationPolluter(BasePolluter):
    def __init__(self, min_val: Optional[float] = None, max_val: Optional[float] = None):
        self.min_val = min_val
        self.max_val = max_val
        super().__init__(self.apply_minmax)

    def apply_minmax(self, df: pl.DataFrame, column: str):
        if self.min_val is None:
            self.min_val = df[column].min()
        if self.max_val is None:
            self.max_val = df[column].max()

        denominator = self.max_val - self.min_val
        if denominator == 0:
            return np.zeros(len(df))
        return (df[column] - self.min_val) / denominator

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Float64}
