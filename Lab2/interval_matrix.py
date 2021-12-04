from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abc import ABC, abstractclassmethod


class Interval:
    @staticmethod
    def intersection(i1: Interval, i2: Interval) -> Interval:
        a1, b1 = i1.interval_boundaries()
        a2, b2 = i2.interval_boundaries()

        a, b = max(a1, a2), min(b1, b2)
        assert a <= b

        return Interval(a, b)

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self._a = a;
        self._b = b;

    def interval_boundaries(self):
        return self._a, self._b

    def copy(self) -> Interval:
        return Interval(self._a, self._b)

    def __add__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(self._a + a, self._b + b)

    def __sub__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(self._a - b, self._b - a)

    def __mul__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(min([self._a * a, self._a * b, self._b * a, self._b * b]),
                        max([self._a * a, self._a * b, self._b * a, self._b * b]))

    def __truediv__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return self * Interval(1 / b, 1/ a)

    def __pow__(self, power: int) -> Interval:
        interval = Interval(1, 1)

        for _ in range(power):
            interval = interval * self

        return interval

    def scale(self, mul: int) -> Interval:
        a, b = self.interval_boundaries()

        a *= mul
        b *= mul

        return Interval(min(a, b), max(a, b))

    def rad(self) -> float:
        return (self._b - self._a) / 2

    def center(self) -> float:
        return (self._a + self._b) / 2

    def abs(self) -> float:
        return max(-self._a, self._b)


class Matrix:
    @staticmethod
    def identity(size: int) -> Matrix:
        res = Matrix(size)

        for i in range(size):
            res[[i, i]] = 1.0

        return res

    def __init__(self, size: int) -> None:
        self.size = size
        self.matrix = np.array([np.array([0.0 for _ in range(size)]) for _ in range(size)])

    def at(self, i: int, j: int) -> float:
        assert 0 <= i < self.size and 0 <= j < self.size

        return self.matrix[i][j]

    def sz(self) -> int:
        return self.size

    def __setitem__(self, ind: list, value: float) -> None:
        assert len(ind) == 2
        i, j = ind[0], ind[1]
        assert 0 <= i < self.size and 0 <= j < self.size

        self.matrix[i][j] = value

    def __getitem__(self, ind: list) -> float:
        assert len(ind) == 2
        i, j = ind[0], ind[1]
        assert 0 <= i < self.size and 0 <= j < self.size

        return self.matrix[i][j]

    def mul_interval_vector(self, vector: IntervalVector) -> IntervalVector:
        assert vector.sz() == self.sz()

        v = IntervalVector(vector.sz())

        for i in range(vector.sz()):
            interval = Interval(0, 0)

            for j in range(vector.sz()):
                interval = interval + vector.at(j).scale(self[[i, j]])

            v.set(i, interval)

        return v

    def interval_matrix(self) -> IntervalMatrix:
        res = DefaultIntervalMatrix(self.sz())

        for i in range(self.sz()):
            for j in range(self.sz()):
                elem = self[[i, j]]
                res.set(i, j, Interval(elem, elem))

        return res

    def copy(self):
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                matrix.matrix[i][j] = self.matrix[i][j]

        return matrix

    def det(self) -> float:
        return np.linalg.det(self.matrix)

    def svd(self) -> np.array(float):
        return np.linalg.svd(self.matrix, compute_uv=False)

    def spectral_radius(self) -> float:
        ws = np.linalg.eigvals(self.matrix)
        return max(np.abs(ws[i]) for i in range(ws.size))

    def inverse(self) -> Matrix:
        res = Matrix(self.sz())
        inv = np.linalg.inv(self.matrix)

        for i in range(self.sz()):
            for j in range(self.sz()):
                res[[i, j]] = inv[i][j]

        return res


class IntervalMatrix(ABC):
    def __init__(self, size: int) -> None:
        self.size = size
        self.matrix = np.array([np.array([Interval(0, 0) for _ in range(size)]) for _ in range(size)])

    def sz(self) -> int:
        return self.size

    @abstractclassmethod
    def at(self, i: int, j: int) -> Interval:
        pass

    def set(self, i: int, j: int, interval: Interval) -> None:
        self.matrix[i][j] = interval.copy()

    def mul_matrix(self, matrix: Matrix) -> IntervalMatrix:
        assert matrix.sz() == self.size

        m = DefaultIntervalMatrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                interval = Interval(0, 0)
                
                for k in range(self.size):
                    interval = interval + self.at(k, j).scale(matrix[[i, k]])
                
                m.set(i, j, interval)

        return m

    def sub_interval_matrix(self, imatrix: IntervalMatrix) -> IntervalMatrix:
        assert self.sz() == imatrix.sz()

        res = DefaultIntervalMatrix(self.sz())

        for i in range(self.sz()):
            for j in range(self.sz()):
                res.set(i, j, self.at(i, j) - imatrix.at(i, j))
        
        return res

    def mul_vector(self, vector: IntervalVector) -> IntervalVector:
        assert vector.sz() == self.size

        v = IntervalVector(self.size)

        for i in range(self.size):
            interval = Interval(0, 0)

            for j in range(self.size):
                interval = interval + self.at(i, j) * vector.at(j)

            v.set(i, interval)

        return v

    def rad_matrix(self) -> Matrix:
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                matrix[[i, j]] = self.at(i, j).rad()

        return matrix

    def mid_matrix(self) -> Matrix:
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                a, b = self.at(i, j).interval_boundaries()
                matrix[[i, j]] = (a + b) / 2

        return matrix

    def abs_matrix(self) -> Matrix:
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                matrix[[i, j]] = self.at(i, j).abs()

        return matrix

    def norm_inf(self) -> float:
        norm = 0.0

        for line in self.matrix:
            s = sum(line[i].abs() for i in range(line.size))
            norm = max(norm, s)
        
        return norm

class DefaultIntervalMatrix(IntervalMatrix):
    def __init__(self, size: int) -> None:
        super().__init__(size)

    def at(self, i: int, j: int) -> Interval:
        return self.matrix[i][j].copy()


class IntervalVector(ABC):
    @staticmethod
    def intersection(iv1: IntervalVector, iv2: IntervalVector) -> IntervalVector:
        size = iv1.sz()
        assert size == iv2.sz()

        iv = IntervalVector(size)
        
        for i in range(size):
            iv.set(i, Interval.intersection(iv1.at(i), iv2.at(i)))

        return iv

    @staticmethod
    def create(intervals: npt.ArrayLike) -> IntervalVector:
        size = intervals.size
        ivector = IntervalVector(size)
        
        for i in range(size):
            ivector.set(i, intervals[i])

        return ivector

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

    def __sub__(self, other: IntervalVector) -> IntervalVector:
        size = self.sz()
        assert size == other.sz()

        return IntervalVector.create(np.array([
            self.at(i) - other.at(i) for i in range(size)
        ]))

    def sz(self) -> int:
        return self.vector.size

    def norm_inf(self) -> float:
        return max([self.vector[i].abs() for i in range(self.sz())])

    def max_rad(self) -> float:
        return max([self.vector[i].rad() for i in range(self.sz())])

    def center_point(self) -> np.array:
        return np.array([self.vector[i].center() for i in range(self.sz())])


class TwoDimmensionalTaskVector(IntervalVector):
    def __init__(self, interval: Interval) -> None:
        super().__init__(2)

        self.vector[0] = interval.copy()
        self.vector[1] = Interval(0, 0)


class TwoDimmensionalTaskMatrix(IntervalMatrix):
    def __init__(self, a: float, b: float, interval: Interval) -> None:
        super().__init__(2)

        self.matrix[0][0] = Interval(a, a)
        self.matrix[0][1] = Interval(b, b)
        self.matrix[1][0] = Interval(1, 1)

        a, b = interval.interval_boundaries()
        assert 0 <= a <= b

        self.matrix[1][1] = Interval(-b, -a)

    def at(self, i: int, j: int) -> Interval:
        return self.matrix[i][j].copy()
