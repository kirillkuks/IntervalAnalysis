from __future__ import annotations

from solver import Solver

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix, Matrix

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class SubdifferentialNewton(Solver):
    @staticmethod
    def sti(ivector: IntervalVector) -> npt.ArrayLike:
        size = ivector.sz()
        vec1 = np.array([0.0 for _ in range(size)])
        vec2 = np.array([0.0 for _ in range(size)])

        for i in range(size):
            a, b = ivector.at(i).interval_boundaries()
            vec1[i] = a
            vec2[i] = b

        return np.append(vec1, vec2)

    @staticmethod
    def sti_inv(vector: npt.ArrayLike) -> IntervalVector:
        size = vector.size
        assert size % 2 == 0

        h_size = int(size / 2)

        intervals = np.array([Interval(vector[i], vector[i + h_size]) for i in range(h_size)])
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
    def zero_matrix2n2n(sz: int) -> Matrix:
        assert sz > 0
        size = 2 * sz

        lines = np.array([np.array([0 for _ in range(size)]) for _ in range(size)])
        return Matrix.create(lines)

    @staticmethod
    def _build_matrix2n2n_elem(value: float, i: int, j: int, size: int) -> float:
        assert 0 <= i < size
        assert 0 <= j < size

        h_size = int(size / 2)

        if (i < h_size and j < h_size) or (i >= h_size and j >= h_size):
            return max(value, 0)

        return -max(-value, 0)

    @staticmethod
    def create(imatrix: IntervalMatrix, ivector: IntervalVector) -> SubdifferentialNewton:
        s1, s2 = imatrix.sz()
        assert s1 == s2
        assert s1 == ivector.sz()

        return SubdifferentialNewton(imatrix, ivector)

    def __init__(self, imatrix: IntervalMatrix, ivector: IntervalVector) -> None:
        super().__init__()
        self.A = imatrix
        self.b = ivector
        self.size = self.b.sz()

        self.max_iters = 100
        self.tau = 0.1

    def solve(self, eps: float = 1e-16) -> IntervalVector:
        C = self.__class__.matrix2n2n(self.A.mid())
        d = self.__class__.sti(self.b)
        print(C.matrix)

        x_k = self.first_approximation(C, d)
        self.x_iters.append(x_k)
        print(x_k.at(0).interval_boundaries())
        print(x_k.at(1).interval_boundaries())

        # Fx_k = C.mul_vector(self.__class__.sti(x_k))
        # print(Fx_k)

        counter = 0
        while counter < self.max_iters:
            counter += 1

            xsti = self.__class__.sti(x_k)
            xxsti = np.copy(xsti)

            F = self.__class__.zero_matrix2n2n(self.size)

            for i in range(self.size):
                s0, s1 = 0, 0 

                for j in range(self.size):
                    g0, g1 = self.A.at(i, j).interval_boundaries()
                    h0, h1 = xxsti[j], xxsti[j + self.size]

                    multy_type = self.define_multy_type(Interval(g0, g1), Interval(h0, h1))
                    assert 0 < multy_type < 17

                    t0, t1 = 0, 0
                    if multy_type == 1:
                        t0, t1 = g0 * h0, g1 * h1
                        F.set(i, j, g0)
                        F.set(i + self.size, j + self.size, g1)
                    elif multy_type == 2:
                        t0, t1 = g1 * h0, g1 * h1
                        F.set(i, j, g1)
                        F.set(i + self.size, j + self.size, g1)
                    elif multy_type == 3:
                        t0, t1 = g1 * h0, g0 * h1
                        F.set(i, j, g1)
                        F.set(i + self.size, j + self.size, g0)
                    elif multy_type == 4:
                        t0, t1 = g0 * h0, g0 * h1
                        F.set(i, j, g0)
                        F.set(i + self.size, j + self.size, g0)
                    elif multy_type == 5:
                        t0, t1 = g0 * h1, g1 * h1
                        F.set(i, j + self.size, g0)
                        F.set(i + self.size, j + self.size, g1)
                    elif multy_type == 6:
                        u0, u1 = g0 * h1, g0 * h0
                        v0, v1 = g1 * h0, g1 * h1
                        if u0 < v0:
                            t0 = u0
                            F.set(i, j + self.size, g0)
                        else:
                            t0 = v0
                            F.set(i, j, g1)
                        if u1 > v1:
                            t1 = u1
                            F.set(i + self.size, j, g0)
                        else:
                            t1 = v1
                            F.set(i + self.size, j + self.size, g1)
                    elif multy_type == 7:
                        t0, t1 = g1 * h0, g0 * h0
                        F.set(i, j, g1)
                        F.set(i + self.size, j, g0)
                    elif multy_type == 8:
                        t0, t1 = 0, 0
                    elif multy_type == 9:
                        t0, t1 = g0 * h1, g1 * h0
                        F.set(i, j + self.size, g0)
                        F.set(i + self.size, j, g1)
                    elif multy_type == 10:
                        t0, t1 = g0 * h1, g0 * h0
                        F.set(i, j + self.size, g0)
                        F.set(i + self.size, j, g0)
                    elif multy_type == 11:
                        t0, t1 = g1 * h1, g0 * h0
                        F.set(i, j + self.size, g1)
                        F.set(i + self.size, j, g0)
                    elif multy_type == 12:
                        t0, t1 = g1 * h1, g1 * h0
                        F.set(i, j + self.size, g1)
                        F.set(i + self.size, j, g1)
                    elif multy_type == 13:
                        t0, t1 = g0 * h0, g1 * h0
                        F.set(i, j, g0)
                        F.set(i + self.size, j, g1)
                    elif multy_type == 14:
                        t0, t1 = 0, 0
                    elif multy_type == 15:
                        t0, t1 = g1 * h1, g0 * h1
                        F.set(i, j + self.size, g1)
                        F.set(i + self.size, j + self.size, g0)
                    elif multy_type == 16:
                        u0, u1 = g0 * h0, g0 * h1
                        v0, v1 = g1 * h1, g1 * h0
                        if u0 > v0:
                            t0 = u0
                            F.set(i, j, g0)
                        else:
                            t0 = v0
                            F.set(i, j + self.size, g1)
                        if u1 < v1:
                            t1 = u1
                            F.set(i + self.size, j + self.size, g0)
                        else:
                            t1 = v1
                            F.set(i + self.size, j, g1)

                    s0 += t0
                    s1 += t1
                
                d0, d1 = self.b.at(i).interval_boundaries()
                q0 = s0 - d0
                q1 = s1 - d1
                xsti[i] = q0
                xsti[i + self.size] = q1

            x_k = self.__class__.sti_inv(xxsti - self.tau * np.linalg.solve(F.data(), xsti))
            self.x_iters.append(x_k)

            if IntervalVector.diff(self.x_iters[counter], self.x_iters[counter - 1]) < eps:
                break

            print(f'Iter: {counter}')
            print(x_k.at(0).interval_boundaries())
            print(x_k.at(1).interval_boundaries())
                    
        print(f'Iters num = {counter}')
        print('Result:')
        print(x_k.at(0).interval_boundaries())
        print(x_k.at(1).interval_boundaries())

        # self.x_iters = self.x_iters[90:]
        self.__class__.ave(self.x_iters[90:])

        self._plot_interval_2Dbars()
        self._plot_bars_radiuses()

        return x_k

    @staticmethod
    def ave(xs: list) -> IntervalVector:
        iv = IntervalVector.create(np.array([Interval(0, 0), Interval(0, 0)]))
        for i in range(len(xs)):
            iv = iv + xs[i]

        for i in range(iv.sz()):
            a, b = iv.at(i).interval_boundaries()
            print(f'AVE: {a / len(xs)} | {b / len(xs)}') 

    def first_approximation(self, C, d) -> IntervalVector:
        x = np.linalg.solve(C.data(), d)
        return self.__class__.sti_inv(x)

    def define_multy_type(self, i1: Interval, i2: Interval) -> int:
        g0, g1 = i1.interval_boundaries()
        h0, h1 = i2.interval_boundaries()

        l, m = 0, 0

        if g0 * g1 > 0:
            if g0 > 0:
                l = 0
            else:
                l = 2
        else:
            if g0 <= g1:
                l = 1
            else:
                l = 3
        if h0 * h1 > 0:
            if h0 > 0:
                m = 1
            else:
                m = 3
        else:
            if h0 <= h1:
                m = 2
            else:
                m = 4

        return 4 * l + m
