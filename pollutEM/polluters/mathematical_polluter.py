from typing import Union
import numpy as np

from .base_polluter import BasePolluter


class ScalingPolluter(BasePolluter):
    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        super().__init__(self.apply_linear_transform)

    def apply_linear_transform(self, value):
        return value * self.multiplier

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.int64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


class ShiftingPolluter(BasePolluter):
    def __init__(self, shift_amount: Union[int, float]):
        self.shift_amount = shift_amount
        super().__init__(self.apply_shift)

    def apply_shift(self, value):
        return value + self.shift_amount

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.int64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


class ReciprocalPolluter(BasePolluter):
    def __init__(self):
        super().__init__(self.apply_reciprocal)

    def apply_reciprocal(self, value):
        return 1.0 / value if value != 0 else float("inf")

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


class LogTransformationPolluter(BasePolluter):
    def __init__(self, base: float = np.e, handle_zeros: bool = True):
        self.base = base
        self.handle_zeros = handle_zeros
        super().__init__(self.apply_log)

    def apply_log(self, value):
        if self.handle_zeros and value <= 0:
            value = np.finfo(float).eps
        return float(np.log(value) / np.log(self.base))

    def _get_type_mapping(self) -> dict:
        return {
            int: float,
            float: float,
            np.float64: np.float64,
            np.int64: np.float64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


# AffineTransformationPolluter
# ExponentialTransformationPolluter
#
