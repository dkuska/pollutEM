import numpy as np

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

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: np.float64,
            np.int64: np.float64,
        }


# UniformNoisePolluter
# SaltAndPepperNoisePolluter
# MissingValuesPolluter
# IntegerOverflowPolluter
