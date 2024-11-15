from .error_polluter import GaussianNoisePolluter, UniformNoisePolluter
from .mathematical_polluter import (
    ScalingPolluter,
    ShiftingPolluter,
    LogTransformationPolluter,
    ReciprocalPolluter,
)
from .precision_polluter import RoundingPolluter
from .representation_polluter import ScientificNotationPolluter, SeparatorConversionPolluter

__all__ = [
    "GaussianNoisePolluter",
    "LogTransformationPolluter",
    "ReciprocalPolluter",
    "RoundingPolluter",
    "ScalingPolluter",
    "ScientificNotationPolluter",
    "SeparatorConversionPolluter",
    "ShiftingPolluter",
    "UniformNoisePolluter",
]


def get_polluter(polluter_name, **kwargs):
    if polluter_name == "GaussianNoise":
        return GaussianNoisePolluter(**kwargs)
    elif polluter_name == "LogTransformationPolluter":
        return LogTransformationPolluter(**kwargs)
    elif polluter_name == "ReciprocalPolluter":
        return ReciprocalPolluter(**kwargs)
    elif polluter_name == "RoundingPolluter":
        return RoundingPolluter(**kwargs)
    elif polluter_name == "ScalingPolluter":
        return ScalingPolluter(**kwargs)
    elif polluter_name == "ScientificNotationPolluter":
        return ScientificNotationPolluter(**kwargs)
    elif polluter_name == "SeparatorConversionPolluter":
        return SeparatorConversionPolluter(**kwargs)
    elif polluter_name == "ShiftingPolluter":
        return ShiftingPolluter(**kwargs)
    elif polluter_name == "UniformNoisePolluter":
        return UniformNoisePolluter(**kwargs)
    else:
        raise ValueError(f"No Polluter of Name {polluter_name} known")
