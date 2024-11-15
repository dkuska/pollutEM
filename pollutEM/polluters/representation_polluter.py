import numpy as np
import pandas as pd
from typing import Union

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

    def _get_type_mapping(self) -> dict:
        # Since we're converting numbers to scientific notation strings
        return {
            np.float64: str,
            np.int64: str,
            str: str,
            object: str,
            pd.StringDtype: pd.StringDtype,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


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

    def _get_type_mapping(self) -> dict:
        return {
            np.float64: str,
            np.int64: str,
            str: str,
            object: str,
            pd.StringDtype: pd.StringDtype,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["cell", "row", "column"]


# Roman Numeral Conversion

# Number Word Conversion
