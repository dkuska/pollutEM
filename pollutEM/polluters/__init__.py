import pandas as pd
import sys
from typing import Any

from .domain_specific_polluter import CoordinatePolluter, TimestampPolluter
from .error_polluter import GaussianNoisePolluter, MissingValuePolluter, UniformNoisePolluter
from .mathematical_polluter import ScalingPolluter, ShiftingPolluter, ScalingShiftingPolluter
from .pollution_config_generator import PollutionConfigGenerator
from .precision_polluter import RoundingPolluter

__all__ = [
    "CoordinatePolluter",
    "GaussianNoisePolluter",
    "MissingValuePolluter",
    "PollutionConfigGenerator",
    "RoundingPolluter",
    "ScalingPolluter",
    "ScalingShiftingPolluter",
    "ShiftingPolluter",
    "TimestampPolluter",
    "UniformNoisePolluter",
]


def get_polluter(polluter_name, **kwargs):
    if polluter_name == "Coordinate":
        return CoordinatePolluter(**kwargs)
    elif polluter_name == "GaussianNoise":
        return GaussianNoisePolluter(**kwargs)
    elif polluter_name == "MissingValue":
        return MissingValuePolluter(**kwargs)
    elif polluter_name == "Rounding":
        return RoundingPolluter(**kwargs)
    elif polluter_name == "Scaling":
        return ScalingPolluter(**kwargs)
    elif polluter_name == "ScalingShifting":
        return ScalingShiftingPolluter(**kwargs)
    elif polluter_name == "Shifting":
        return ShiftingPolluter(**kwargs)
    elif polluter_name == "Timestamp":
        return TimestampPolluter(**kwargs)
    elif polluter_name == "UniformNoise":
        return UniformNoisePolluter(**kwargs)
    else:
        raise ValueError(f"No Polluter of Name {polluter_name} known")


def apply_pollutions(df: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Apply the specified pollutions to the dataset."""
    try:
        polluted_df = df.copy()

        for pollution_params in config.get("pollutions", []):
            polluter = get_polluter(
                polluter_name=pollution_params["name"], **pollution_params["params"]
            )
            polluted_df = polluter.apply(polluted_df)

        return polluted_df
    except Exception:
        sys.exit(1)
