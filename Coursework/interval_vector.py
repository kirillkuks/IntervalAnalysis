from __future__ import annotations

from interval import Interval

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

class IntervalVector(ABC):
    @staticmethod
    def create(intervals: npt.ArrayLike) -> IntervalVector:
        ivector = IntervalVector(intervals.size)

        for i in range(intervals.size):
            ivector.set(i, intervals[i])

        return ivector

    @staticmethod
    def intersection(iv1: IntervalVector, iv2: IntervalVector) -> IntervalVector:
        size = iv1.sz()
        assert size == iv2.sz()

        iv = IntervalVector(size)
        
        for i in range(size):
            iv.set(i, Interval.intersection(iv1.at(i), iv2.at(i)))

        return iv

    def __init__(self, size: int) -> None:
        assert size > 0

        self.vector = np.array([Interval() for _ in range(size)])

    def at(self, ind: int) -> Interval:
        assert 0 <= ind < self.vector.size
        
        return self.vector[ind].copy()

    def set(self, ind: int, interval: Interval) -> None:
        assert 0 <= ind < self.vector.size

        self.vector[ind] = interval.copy()

    def __add__(self, other: IntervalVector) -> IntervalVector:
        size = self.sz()
        assert size == other.sz()

        iv = IntervalVector(size)

        for i in range(size):
            iv.set(i, self.at(i) + other.at(i))

        return iv

    def sz(self) -> int:
        return self.vector.size

    def norm_inf(self) -> float:
        return max([self.vector[i].abs() for i in range(self.sz())])

    def norm(self) -> float:
        return np.sqrt(np.sum(np.array([
            interval.abs() ** 2 for interval in self.vector
        ])))

    def max_rad(self) -> float:
        return max([self.vector[i].rad() for i in range(self.sz())])

    def center_point(self) -> np.array:
        return np.array([self.vector[i].center() for i in range(self.sz())])