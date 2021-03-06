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

# class Matrix:
#     @staticmethod
#     def identity(size: int) -> Matrix:
#         res = Matrix(size)

#         for i in range(size):
#             res[[i, i]] = 1.0

#         return res

#     def __init__(self, size: int) -> None:
#         self.size = size
#         self.matrix = np.array([np.array([0.0 for _ in range(size)]) for _ in range(size)])

#     def at(self, i: int, j: int) -> float:
#         assert 0 <= i < self.size and 0 <= j < self.size

#         return self.matrix[i][j]

#     def sz(self) -> int:
#         return self.size

#     def __setitem__(self, ind: list, value: float) -> None:
#         assert len(ind) == 2
#         i, j = ind[0], ind[1]
#         assert 0 <= i < self.size and 0 <= j < self.size

#         self.matrix[i][j] = value

#     def __getitem__(self, ind: list) -> float:
#         assert len(ind) == 2
#         i, j = ind[0], ind[1]
#         assert 0 <= i < self.size and 0 <= j < self.size

#         return self.matrix[i][j]

#     def mul_interval_vector(self, vector: IntervalVector) -> IntervalVector:
#         assert vector.sz() == self.sz()

#         v = IntervalVector(vector.sz())

#         for i in range(vector.sz()):
#             interval = Interval(0, 0)

#             for j in range(vector.sz()):
#                 interval = interval + vector.at(j).scale(self[[i, j]])

#             v.set(i, interval)

#         return v

#     def interval_matrix(self) -> IntervalMatrix:
#         res = DefaultIntervalMatrix(self.sz())

#         for i in range(self.sz()):
#             for j in range(self.sz()):
#                 elem = self[[i, j]]
#                 res.set(i, j, Interval(elem, elem))

#         return res

#     def copy(self):
#         matrix = Matrix(self.size)

#         for i in range(self.size):
#             for j in range(self.size):
#                 matrix.matrix[i][j] = self.matrix[i][j]

#         return matrix

#     def det(self) -> float:
#         return np.linalg.det(self.matrix)

#     def svd(self) -> np.array(float):
#         return np.linalg.svd(self.matrix, compute_uv=False)

#     def spectral_radius(self) -> float:
#         ws = np.linalg.eigvals(self.matrix)
#         return max(np.abs(ws[i]) for i in range(ws.size))

#     def inverse(self) -> Matrix:
#         res = Matrix(self.sz())
#         inv = np.linalg.inv(self.matrix)

#         for i in range(self.sz()):
#             for j in range(self.sz()):
#                 res[[i, j]] = inv[i][j]

#         return res

# class IntervalMatrix(ABC):
#     def __init__(self, size: int) -> None:
#         self.size = size
#         self.matrix = np.array([np.array([Interval(0, 0) for _ in range(size)]) for _ in range(size)])

#     def sz(self) -> int:
#         return self.size

#     @abstractmethod
#     def at(self, i: int, j: int) -> Interval:
#         pass

#     def set(self, i: int, j: int, interval: Interval) -> None:
#         self.matrix[i][j] = interval.copy()

#     def mul_matrix(self, matrix: Matrix) -> IntervalMatrix:
#         assert matrix.sz() == self.size

#         m = DefaultIntervalMatrix(self.size)

#         for i in range(self.size):
#             for j in range(self.size):
#                 interval = Interval(0, 0)
                
#                 for k in range(self.size):
#                     interval = interval + self.at(k, j).scale(matrix[[i, k]])
                
#                 m.set(i, j, interval)

#         return m

#     def sub_interval_matrix(self, imatrix: IntervalMatrix) -> IntervalMatrix:
#         assert self.sz() == imatrix.sz()

#         res = DefaultIntervalMatrix(self.sz())

#         for i in range(self.sz()):
#             for j in range(self.sz()):
#                 res.set(i, j, self.at(i, j) - imatrix.at(i, j))
        
#         return res

#     def mul_vector(self, vector: IntervalVector) -> IntervalVector:
#         assert vector.sz() == self.size

#         v = IntervalVector(self.size)

#         for i in range(self.size):
#             interval = Interval(0, 0)

#             for j in range(self.size):
#                 interval = interval + self.at(i, j) * vector.at(j)

#             v.set(i, interval)

#         return v

#     def rad_matrix(self) -> Matrix:
#         matrix = Matrix(self.size)

#         for i in range(self.size):
#             for j in range(self.size):
#                 matrix[[i, j]] = self.at(i, j).rad()

#         return matrix

#     def mid_matrix(self) -> Matrix:
#         matrix = Matrix(self.size)

#         for i in range(self.size):
#             for j in range(self.size):
#                 a, b = self.at(i, j).interval_boundaries()
#                 matrix[[i, j]] = (a + b) / 2

#         return matrix

#     def abs_matrix(self) -> Matrix:
#         matrix = Matrix(self.size)

#         for i in range(self.size):
#             for j in range(self.size):
#                 matrix[[i, j]] = self.at(i, j).abs()

#         return matrix

#     def norm_inf(self) -> float:
#         norm = 0.0

#         for line in self.matrix:
#             s = sum(line[i].abs() for i in range(line.size))
#             norm = max(norm, s)
        
#         return norm

# class DefaultIntervalMatrix(IntervalMatrix):
#     def __init__(self, size: int) -> None:
#         super().__init__(size)

#     def at(self, i: int, j: int) -> Interval:
#         return self.matrix[i][j].copy()
