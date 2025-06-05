from .size import Size

from dataclasses import dataclass
from typing import Mapping


@dataclass(slots=True, frozen=True)
class SprintIntake:
    totals: Mapping[Size, int]
    hours: float