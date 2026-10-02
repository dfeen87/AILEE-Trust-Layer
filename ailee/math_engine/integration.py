"""Small deterministic numerical integration primitives."""

import math
from typing import Optional, Sequence, Tuple

from .validation import MathNumericalError, MathValidationError, finite_number


def trapezoidal_integral(points: Sequence[Tuple[float, float]]) -> float:
    """Integrate ordered ``(time, value)`` points with composite trapezoids."""
    if isinstance(points, (str, bytes)) or not isinstance(points, Sequence):
        raise MathValidationError("points must be an ordered sequence")
    if len(points) < 2:
        raise MathValidationError("at least two integration points are required")
    total = 0.0
    previous_time: Optional[float] = None
    previous_value: Optional[float] = None
    for index, point in enumerate(points):
        if not isinstance(point, (tuple, list)) or len(point) != 2:
            raise MathValidationError(f"points[{index}] must contain (time, value)")
        time = finite_number(point[0], f"points[{index}].time")
        value = finite_number(point[1], f"points[{index}].value")
        if previous_time is not None and previous_value is not None:
            if time <= previous_time:
                raise MathValidationError(
                    "integration times must be strictly increasing"
                )
            area = 0.5 * (previous_value + value) * (time - previous_time)
            if not math.isfinite(area) or not math.isfinite(total + area):
                raise MathNumericalError(
                    "trapezoidal integration produced a non-finite value"
                )
            total += area
        previous_time, previous_value = time, value
    return total
