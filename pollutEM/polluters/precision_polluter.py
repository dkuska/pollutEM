import numpy as np

from .base_polluter import BasePolluter


class RoundingPolluter(BasePolluter):
    # TODO: Add different strategies: Rounding, Floor, Ceiling
    def __init__(self, decimal_places: int = 0):
        self.decimal_places = decimal_places
        super().__init__(self.round_transform)

    def round_transform(self, value):
        if isinstance(value, float):
            return round(value, self.decimal_places)
        return value

    def _get_type_mapping(self) -> dict:
        # Rounding preserves the original numeric type
        return {
            np.float64: np.float64,
            np.int64: np.int64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]
