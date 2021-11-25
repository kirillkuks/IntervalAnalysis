from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Interval:
    @staticmethod
    def create(a: float, b: float) -> Interval:
        return Interval(a, b)

    def __init__(self, a: float, b: float) -> None:
        self.a = min(a, b)
        self.b = max(a, b)

    def __add__(self, other: Interval) -> Interval:
        return Interval(self.a + other.a, self.b + other.b)

    def __sub__(self, other: Interval) -> Interval:
        return Interval(self.a - other.b, self.b - other.a)

    def __mul__(self, other: Interval) -> Interval:
        return Interval(
            min(self.a * other.a, self.a * other.b, self.b * other.a, self.b * other.b),
            max(self.a * other.a, self.a * other.b, self.b * other.a, self.b * other.b)
        )

    def __pow__(self, power: int) -> Interval:
        interval = Interval(1, 1)

        for _ in range(power):
            interval = interval * self

        return interval

    def scale(self, factor: float) -> Interval:
        if factor >= 0:
            return Interval(factor * self.a, factor * self.b)

        return Interval(factor * self.b, factor * self.a)

    def add_scalar(self, num: float) -> Interval:
        return Interval(self.a + num, self.b + num)

    def cos(self) -> Interval:
        a = np.cos(self.a)
        b = np.cos(self.b)

        return Interval(min(a, b), max(a, b))

    def dist(self, p: float) -> float:
        if self.a <= p <= self.b:
            return 0

        return min(np.abs(p - self.a), np.abs(p - self.b))

    def max(self) -> float:
        return max(self.a, self.b)

    def min(self) -> float:
        return min(self.a, self.b)

    def rad(self) -> float:
        return (self.b - self.a) / 2

    def mid(self) -> float:
        return (self.a + self.b) / 2

    def copy(self) -> Interval:
        return Interval(self.a, self.b)
