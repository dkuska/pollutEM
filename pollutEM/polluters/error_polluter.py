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


class IntegerOverflowPolluter(BasePolluter):
    def __init__(
        self,
        threshold: int = 2**16 - 1,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.threshold = threshold
        super().__init__(
            transformation=self.simulate_overflow,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def simulate_overflow(self, value):
        if isinstance(value, pd.Series):
            return value.apply(self.simulate_overflow)
        if isinstance(value, (int, np.int64)):
            if abs(value) >= self.threshold:
                sign = -1 if value > 0 else 1
                return sign * (abs(value) % (self.threshold + 1))
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.int64: np.int64,
            np.float64: np.int64,
        }


class DigitPermutationPolluter(BasePolluter):
    def __init__(
        self,
        num_swaps: int = 1,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        if num_swaps < 1:
            raise ValueError("num_swaps must be at least 1")
        self.num_swaps = num_swaps
        super().__init__(
            transformation=self.permute_digits,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def permute_digits(self, value):
        if isinstance(value, (int, float)):
            str_val = str(abs(value))
            digits = list(str_val.replace(".", ""))

            if len(digits) >= 2:
                for _ in range(self.num_swaps):
                    i, j = np.random.choice(len(digits), 2, replace=False)
                    digits[i], digits[j] = digits[j], digits[i]

                # Reconstruct number with original type
                result = "".join(digits)
                if isinstance(value, float) or "." in str_val:
                    decimal_pos = str_val.find(".")
                    if decimal_pos != -1:
                        result = result[:decimal_pos] + "." + result[decimal_pos:]
                    result = np.float64(result)
                else:
                    result = np.int64(result)

                return -result if value < 0 else result
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.int64: np.int64,
            np.float64: np.float64,
        }


class BitShiftPolluter(BasePolluter):
    def __init__(
        self,
        direction: str = "left",  # "left" or "right"
        num_bits: int = 1,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        if direction not in ["left", "right"]:
            raise ValueError("direction must be either 'left' or 'right'")
        if num_bits < 1:
            raise ValueError("num_bits must be at least 1")

        self.direction = direction
        self.num_bits = num_bits
        super().__init__(
            transformation=self.shift_bits,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def shift_bits(self, value):
        if isinstance(value, pd.Series):
            return value.apply(self.shift_bits)

        if isinstance(value, (int, np.int64, np.int32)):
            shifted_value = int(value)  # Convert numpy int to Python int
            if self.direction == "left":
                return np.int64(shifted_value << self.num_bits)
            return np.int64(shifted_value >> self.num_bits)

        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.int64: np.int64,
        }


class SignFlipPolluter(BasePolluter):
    def __init__(
        self,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(
            transformation=self.flip_sign,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def flip_sign(self, value):
        if isinstance(value, pd.Series):
            return value.apply(self.flip_sign)
        if isinstance(value, (int, float, np.int64, np.float64)):
            return -value
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.int64: np.int64,
            np.float64: np.float64,
        }


class ModuloPolluter(BasePolluter):
    def __init__(
        self,
        modulo: int = 100,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.modulo = modulo
        super().__init__(
            transformation=self.apply_modulo,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def apply_modulo(self, value):
        if isinstance(value, pd.Series):
            return value.apply(self.apply_modulo)
        if isinstance(value, (int, np.int64)):
            return np.int64(value % self.modulo)
        if isinstance(value, (float, np.float64)):
            return np.float64(int(value) % self.modulo)
        return value

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.int64: np.int64,
            np.float64: np.float64,
        }
