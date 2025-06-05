from enum import Enum

class Size(Enum):
    XS = 2.0
    S = 4.0
    M = 8.0
    L = 16.0
    XL = 24.0
    XXL = float("inf")

    @classmethod
    def classify(cls, hrs: float) -> "Size":
        for s in cls:
            if hrs <= s.value:
                return s
        return cls.XXL
