from typing import Type
import numpy as np
import polars as pl

from .base_polluter import BasePolluter


class LinearTransformationPolluter(BasePolluter):
    def __init__(self, multiplier: float):
        self.multiplier = multiplier
        super().__init__(self.apply_linear_transform)

    def apply_linear_transform(self, value):
        return value * self.multiplier

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Int64}


class ShiftingPolluter(BasePolluter):
    def __init__(self, shift_amount: Union[int, float]):
        self.shift_amount = shift_amount
        super().__init__(self.apply_shift)

    def apply_shift(self, value):
        return value + self.shift_amount

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Int64}


class ExponentiationPolluter(BasePolluter):
    def __init__(self, power: float):
        self.power = power
        super().__init__(self.apply_exponent)

    def apply_exponent(self, value):
        return float(value**self.power)

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Float64}


class ReciprocalPolluter(BasePolluter):
    def __init__(self):
        super().__init__(self.apply_reciprocal)

    def apply_reciprocal(self, value):
        return 1.0 / value if value != 0 else float("inf")

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Float64}


class LogTransformationPolluter(BasePolluter):
    def __init__(self, base: float = np.e, handle_zeros: bool = True):
        self.base = base
        self.handle_zeros = handle_zeros
        super().__init__(self.apply_log)

    def apply_log(self, value):
        if self.handle_zeros and value <= 0:
            value = np.finfo(float).eps
        return float(np.log(value) / np.log(self.base))

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Float64, pl.Int64: pl.Float64}
