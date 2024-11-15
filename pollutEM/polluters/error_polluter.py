from typing import Type
import numpy as np
import polars as pl

from .base_polluter import BasePolluter


class GaussianNoisePolluter(BasePolluter):
    def __init__(self, mean: float = 0, std_dev: float = 1):
        self.mean = mean
        self.std_dev = std_dev
        super().__init__(self.add_gaussian_noise)

    def add_gaussian_noise(self, value):
        if isinstance(value, (int, float)):
            return value + np.random.normal(self.mean, self.std_dev)
        return value

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        # Since adding Gaussian noise produces floating point results
        return {
            pl.Int64: pl.Float64,  # integers become floats after adding noise
            pl.Float64: pl.Float64,  # floats remain floats
        }
