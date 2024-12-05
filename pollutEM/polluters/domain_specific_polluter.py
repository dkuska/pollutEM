import numpy as np
import pandas as pd
from typing import Optional, Union, Type

from .base_polluter import BasePolluter


class TimestampPolluter(BasePolluter):
    def __init__(
        self,
        max_time_shift: int = 3600,  # Maximum time shift in seconds
        direction: str = "both",  # "forward", "backward", or "both"
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        if direction not in ["forward", "backward", "both"]:
            raise ValueError("direction must be 'forward', 'backward', or 'both'")

        self.max_time_shift = max_time_shift
        self.direction = direction
        super().__init__(
            transformation=self.shift_timestamp,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def shift_timestamp(self, value):
        if not isinstance(value, pd.Series):
            return value

        timestamps = pd.to_datetime(value)

        if self.direction == "both":
            shifts = np.random.uniform(-self.max_time_shift, self.max_time_shift, len(value))
        elif self.direction == "forward":
            shifts = np.random.uniform(0, self.max_time_shift, len(value))
        else:  # backward
            shifts = np.random.uniform(-self.max_time_shift, 0, len(value))

        return timestamps + pd.to_timedelta(shifts, unit="s")

    def _get_type_mapping(self) -> dict[Type, Type]:
        return {
            np.datetime64: np.datetime64,
            np.string_: np.string_,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["column"]


class CoordinatePolluter(BasePolluter):
    def __init__(
        self,
        max_deviation: float = 0.001,
        level: str = "column",
        probability: Optional[float] = None,
        indices: Optional[Union[list[int], list[str]]] = None,
        seed: Optional[int] = None,
    ):
        self.max_deviation = max_deviation
        super().__init__(
            transformation=self.shift_coordinates,
            level=level,
            probability=probability,
            indices=indices,
            seed=seed,
        )

    def shift_coordinates(self, value):
        if not isinstance(value, pd.Series):
            return value

        coordinates = pd.to_numeric(value, errors="coerce")
        shifts = np.random.uniform(-self.max_deviation, self.max_deviation, len(value))

        # Ensure latitude stays within [-90, 90]
        if self.is_latitude(value.name):
            result = coordinates + shifts
            return np.clip(result, -90, 90)

        return coordinates + shifts

    def is_latitude(self, column_name: str) -> bool:
        """Identify latitude columns by name"""
        lat_keywords = ["lat", "latitude"]
        return any(keyword in str(column_name).lower() for keyword in lat_keywords)

    def _get_type_mapping(self) -> dict[Type, Type]:
        return {
            np.float64: np.float64,
        }

    def _get_allowed_levels(self) -> list[str]:
        return ["column"]
