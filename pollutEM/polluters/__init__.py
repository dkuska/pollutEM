from .error_polluter import GaussianNoisePolluter, UniformNoisePolluter
from .mathematical_polluter import ScalingPolluter, ShiftingPolluter, ScalingShiftingPolluter
from .precision_polluter import RoundingPolluter

__all__ = [
    "GaussianNoisePolluter",
    "RoundingPolluter",
    "ScalingPolluter",
    "ScalingShiftingPolluter",
    "ShiftingPolluter",
    "UniformNoisePolluter",
]


def get_polluter(polluter_name, **kwargs):
    if polluter_name == "GaussianNoise":
        return GaussianNoisePolluter(**kwargs)
    elif polluter_name == "Rounding":
        return RoundingPolluter(**kwargs)
    elif polluter_name == "Scaling":
        return ScalingPolluter(**kwargs)
    elif polluter_name == "ScalingShifting":
        return ScalingShiftingPolluter(**kwargs)
    elif polluter_name == "Shifting":
        return ShiftingPolluter(**kwargs)
    elif polluter_name == "UniformNoise":
        return UniformNoisePolluter(**kwargs)
    else:
        raise ValueError(f"No Polluter of Name {polluter_name} known")
