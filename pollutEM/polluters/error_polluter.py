from typing import Any, Union, Optional, Type

import numpy as np
import pandas as pd

from .base_polluter import BasePolluter


class GaussianNoisePolluter(BasePolluter):
    def __init__(
        self,
        mean: float = 0.0,
        std_dev: float = 1.0,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.mean = mean
        self.std_dev = std_dev

        super().__init__(
            transformation=self.add_gaussian_noise,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def add_gaussian_noise(self, value):
        if isinstance(value, (int, float)):
            return value + np.random.normal(self.mean, self.std_dev)
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }


class UniformNoisePolluter(BasePolluter):
    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.low = low
        self.high = high

        super().__init__(
            transformation=self.add_uniform_noise,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def add_uniform_noise(self, value):
        if isinstance(value, (int, float)):
            return value + np.random.uniform(self.low, self.high)
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }


class MissingValuePolluter(BasePolluter):
    def __init__(
        self,
        missing_value: Any = np.nan,
        missing_ratio: float = 0.1,  # New parameter to control missing value ratio
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        if not 0 <= missing_ratio <= 1:
            raise ValueError("missing_ratio must be between 0 and 1")

        self.missing_value = missing_value
        self.missing_ratio = missing_ratio
        super().__init__(
            transformation=self.insert_missing,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def insert_missing(self, value):
        if isinstance(value, pd.Series):
            # Calculate number of values to replace
            n_values = len(value)
            n_missing = int(n_values * self.missing_ratio)

            # Create mask for values to replace
            mask = np.zeros(n_values, dtype=bool)
            mask[:n_missing] = True
            np.random.shuffle(mask)

            # Create new series with missing values
            result = value.copy()
            result[mask] = self.missing_value
            return result

        elif isinstance(value, np.ndarray):
            # Similar handling for numpy arrays
            n_values = len(value)
            n_missing = int(n_values * self.missing_ratio)

            mask = np.zeros(n_values, dtype=bool)
            mask[:n_missing] = True
            np.random.shuffle(mask)

            result = value.copy()
            result[mask] = self.missing_value
            return result

        # For single values (cell-level), use probability
        if np.random.random() < self.missing_ratio:
            if isinstance(value, (int, float)) and pd.isna(self.missing_value):
                # Convert integer or float to np.nan directly
                return np.nan
            return self.missing_value
        # For non-missing values, convert to the right type if needed
        if isinstance(value, int) and np.dtype("int64") in self._get_type_mapping():
            return np.float64(value)  # Convert integers to float64 if that's the target type
        return value

    def _get_type_mapping(self) -> dict[Type, Type]:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["row", "column"]
