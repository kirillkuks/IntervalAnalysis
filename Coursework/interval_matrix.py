
from __future__ import annotations

from interval import Interval
from interval_vector import IntervalVector

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt


class IntervalMatrix(ABC):
    @staticmethod
    def create(lines: npt.ArrayLike) -> IntervalMatrix:
        line_length = lines[0].sz()

        for line in lines:
            assert line_length == line.sz()

        return IntervalMatrix(lines)

    def __init__(self, lines: npt.ArrayLike) -> None:
        self.matrix = lines

    def at(self, i: int, j: int) -> Interval:
        assert 0 <= i < self.matrix.size
        assert 0 <= j < self.matrix[i].sz()

        return self.matrix[i].at(j)

    def sz(self):
        return self.matrix.size, self.matrix[0].sz()

    def kaucher_sub(self, other: IntervalMatrix) -> IntervalMatrix:
        lines, columns = other.sz()

        assert self.matrix.size == lines
        assert self.matrix[0].sz() == columns

        return IntervalMatrix.create(
            np.array([
                IntervalVector.create(
                    np.array([
                    Interval.kaucher_sub(self.at(i, j), other.at(i, j))
                for j in range(columns)])
                )
            for i in range(lines)])
        )

    def mid(self) -> npt.ArrayLike:
        return np.array([
            np.array([
                self.at(i, j).center()
            for j in range(self.matrix[i].sz())])
        for i in range(self.matrix.size)])