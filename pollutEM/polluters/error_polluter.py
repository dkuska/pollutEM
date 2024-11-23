from typing import Union, Optional

import numpy as np

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
