from typing import Type
import polars as pl

from .base_polluter import BasePolluter


class RoundingPolluter(BasePolluter):
    def __init__(self, decimal_places: int = 0):
        self.decimal_places = decimal_places
        super().__init__(self.round_transform)

    def round_transform(self, value):
        if isinstance(value, float):
            return round(value, self.decimal_places)
        return value

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        # Rounding preserves the original numeric type
        return {
            pl.Float64: pl.Float64,  # floats remain floats
            pl.Int64: pl.Int64,  # integers remain integers (though rounding won't affect them)
        }
