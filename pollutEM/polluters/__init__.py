from .error_polluter import GaussianNoisePolluter, UniformNoisePolluter
from .mathematical_polluter import (
    ScalingPolluter,
    ShiftingPolluter,
)
from .precision_polluter import RoundingPolluter

__all__ = [
    "GaussianNoisePolluter",
    "RoundingPolluter",
    "ScalingPolluter",
    "ShiftingPolluter",
    "UniformNoisePolluter",
]


def get_polluter(polluter_name, **kwargs):
    if polluter_name == "GaussianNoise":
        return GaussianNoisePolluter(**kwargs)
    elif polluter_name == "RoundingPolluter":
        return RoundingPolluter(**kwargs)
    elif polluter_name == "ScalingPolluter":
        return ScalingPolluter(**kwargs)
    elif polluter_name == "ShiftingPolluter":
        return ShiftingPolluter(**kwargs)
    elif polluter_name == "UniformNoisePolluter":
        return UniformNoisePolluter(**kwargs)
    else:
        raise ValueError(f"No Polluter of Name {polluter_name} known")
