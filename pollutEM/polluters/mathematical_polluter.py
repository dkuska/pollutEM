from typing import Union, Optional
import numpy as np

from .base_polluter import BasePolluter


class ScalingPolluter(BasePolluter):
    def __init__(
        self,
        multiplier: float = 1.0,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int]]] = None,
        seed: Optional[int] = None,
    ):
        self.multiplier = multiplier

        super().__init__(
            transformation=self.apply_linear_transform,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

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
    def __init__(
        self,
        shift_amount: Union[int, float] = 1.0,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.shift_amount = shift_amount
        super().__init__(
            transformation=self.apply_shift,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def apply_shift(self, value):
        return value + self.shift_amount

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.int64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]
