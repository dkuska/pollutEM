from typing import Type
import polars as pl

from .base_polluter import BasePolluter


# Scientific Notation Conversion
class ScientificNotationPolluter(BasePolluter):
    def __init__(self):
        super().__init__(self.scientific_notation_transform)

    @staticmethod
    def scientific_notation_transform(value):
        if isinstance(value, (int, float)):
            return f"{value:.2e}"
        return value

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        # Since we're converting numbers to scientific notation strings
        return {
            pl.Int64: pl.Utf8,  # integers will become strings
            pl.Float64: pl.Utf8,  # floats will become strings
            pl.Utf8: pl.Utf8,  # strings pass through unchanged
        }


# Separator Conversion
class SeparatorConversionPolluter(BasePolluter):
    def __init__(self, decimal_sep: str = ",", thousands_sep: str = "."):
        self.decimal_sep = decimal_sep
        self.thousands_sep = thousands_sep
        super().__init__(self.convert_separators)

    def convert_separators(self, value: Union[float, int, str]) -> str:
        if isinstance(value, (float, int)):
            # Convert number to string with standard format first
            str_val = f"{value:,.2f}"  # Uses default "1,234.56" format
            # Then convert to desired format
            return (
                str_val.replace(",", "@")
                .replace(".", self.decimal_sep)
                .replace("@", self.thousands_sep)
            )
        elif isinstance(value, str):
            try:
                # Try to parse as number and reformat
                num = float(value.replace(",", ""))
                return self.convert_separators(num)
            except ValueError:
                return value
        return str(value)

    def _get_type_mapping(self) -> dict[Type[pl.DataType], Type[pl.DataType]]:
        return {pl.Float64: pl.Utf8, pl.Int64: pl.Utf8, pl.Utf8: pl.Utf8}


# Roman Numeral Conversion

# Number Word Conversion
