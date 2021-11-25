from __future__ import annotations

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix, Matrix

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class SubdifferentialNewton:
    @staticmethod
    def sti(ivector: IntervalVector) -> npt.ArrayLike:
        size = ivector.sz()
        vec1 = np.array([0.0 for _ in range(size)])
        vec2 = np.array([0.0 for _ in range(size)])

        for i in range(size):
            a, b = ivector.at(i).interval_boundaries()
            vec1[i] = -a
            vec2[i] = b

        return np.append(vec1, vec2)

    @staticmethod
    def sti_inv(vector: npt.ArrayLike) -> IntervalVector:
        size = vector.size
        assert size % 2 == 0

        h_size = int(size / 2)

        intervals = np.array([Interval(-vector[i], vector[i + h_size]) for i in range(h_size)])
        return IntervalVector.create(intervals)

    @staticmethod
    def matrix2n2n(matrix: Matrix) -> Matrix:
        sz = matrix.sz()
        size = 2 * sz
        lines = np.array([np.array([SubdifferentialNewton._build_matrix2n2n_elem(matrix[[i % sz, j % sz]], i, j, size)
                                                                                        for j in range(size)])
                                                                                        for i in range(size)])

        return Matrix.create(lines)

    @staticmethod
    def _build_matrix2n2n_elem(value: float, i: int, j: int, size: int) -> float:
        assert 0 <= i < size
        assert 0 <= j < size

        h_size = int(size / 2)

        if (i < h_size and j < h_size) or (i >= h_size and j >= h_size):
            return max(value, 0)

        return max(-value, 0)

    def __init__(self, imatrix: IntervalMatrix, ivector: IntervalVector) -> None:
        self.A = imatrix
        self.b = ivector

    def solve(self) -> IntervalVector:
        C_ = VertC(self.A)
        C = self.__class__.matrix2n2n(self.A.mid())
        d = self.__class__.sti(self.b)

        x_k = self.first_approximation(C, d)

        Fx_k = C.mul_vector(self.__class__.sti(x_k))
        # print(Fx_k)

        return None

    def first_approximation(self, C, d) -> IntervalVector:
        x = np.linalg.solve(C.data(), d)
        return self.__class__.sti_inv(x)


class VertC:
    def __init__(self, imatrix: IntervalMatrix) -> None:
        s1, s2 = imatrix.sz()

        self.line_num = s1
        self.column_num = s2

        self.matrix = np.array([
            np.array([
                self.__class__._build_elem(imatrix.at(i, j))
            for j in range(s2)])
        for i in range(s1)])

        self.max_by_vert(None)

    def max_by_vert(self, x: IntervalVector) -> None:
        i, j = 0, 0

        max_inds = np.array([0 for _ in range(self.line_num * self.column_num)])
        k = 0

        for i in range(self.line_num):
            for j in range(self.column_num):
                max_inds[k] = len(self.matrix[i][j])
                k += 1

        cur_inds = np.array([0 for _ in range(self.line_num * self.column_num)])
        ind = 0

        while True:
            print(cur_inds)
            print(self.matrix_by_ind(cur_inds))

            ind = 0
            while cur_inds[ind] == max_inds[ind] - 1:
                cur_inds[ind] = 0
                ind += 1

                if ind == cur_inds.size:
                    return

            cur_inds[ind] += 1

    def matrix_by_ind(self, ind: npt.ArrayLike) -> np.array:
        return np.array([
            np.array([self.matrix[i][j][ind[2 * i + j]] for j in range(self.column_num)])
        for i in range(self.line_num)])

    @staticmethod
    def _build_elem(interval: Interval) -> list:
        a, b = interval.pro().interval_boundaries()

        if a <= 0 <= b:
            return [a, 0 , b]
        
        return [a, b]
