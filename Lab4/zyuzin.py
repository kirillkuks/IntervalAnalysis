from __future__ import annotations

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import matplotlib.pyplot as plt


class Zyuzin:
    @staticmethod
    def create(imatrix: IntervalMatrix, ivector: IntervalVector) -> Zyuzin:
        s1, s2 = imatrix.sz()
        s = ivector.sz()

        assert s1 == s2 == s

        return Zyuzin(imatrix, ivector)

    def __init__(self, imatrix: IntervalMatrix, ivector: IntervalVector) -> None:
        self.A = imatrix
        self.b = ivector
        self.size = ivector.sz()

        self.max_iters = 10
        self.x_iters = []

    def solve(self) -> IntervalVector:
        assert self.A.is_diag_dominant()

        D = self.create_diag()
        E= self.create_non_diag()
        invD = self.inv_diag_matrix(D)

        x_k = IntervalVector.create(np.array([Interval(-10, 10), Interval(-10, 10)]))
        self.x_iters.append(x_k)

        for _ in range(self.max_iters):
            tmp = self.b.kaucher_sub(E.mul_interval_vector(x_k))
            x_k = invD.mul_interval_vector(tmp)

            self.x_iters.append(x_k)

        print('Result:')
        print(x_k.at(0).interval_boundaries())
        print(x_k.at(1).interval_boundaries())

        self._plot_interval_2Dbars()
        self._plot_bars_radiuses()

        return x_k

    def create_diag(self) -> IntervalMatrix:
        lines = np.array([IntervalVector.create(np.array([self.A.at(i, j).copy() if j == i else Interval(0, 0) for j in range(self.size)]))
                                                                                                      for i in range(self.size)])
        
        return IntervalMatrix.create(lines)

    def create_non_diag(self) -> IntervalMatrix:
        lines = np.array([IntervalVector.create(np.array([self.A.at(i, j).copy() if j != i else Interval(0, 0) for j in range(self.size)]))
                                                                                                      for i in range(self.size)])

        return IntervalMatrix.create(lines)

    def inv_diag_matrix(self, imatrix: IntervalMatrix) -> IntervalMatrix:
        s1, s2 = imatrix.sz()
        assert s1 == s2

        lines = np.array([IntervalVector.create(np.array([imatrix.at(i, j).inv() if i == j else Interval(0, 0) for j in range(s2)])) for i in range(s1)])

        return IntervalMatrix.create(lines)

    def _plot_interval_2Dbars(self) -> None:
        for x_k, i in zip(self.x_iters, range(len(self.x_iters))):
            a, b = x_k.at(0).interval_boundaries()
            c, d = x_k.at(1).interval_boundaries()

            plt.plot(np.array([a, b, b, a, a]), np.array([c, c, d, d, c]), label=f'iter num = {i}')
        
        plt.legend(loc='upper left')
        plt.show()

    def _plot_bars_radiuses(self) -> None:
        sz = len(self.x_iters)

        inter_nums = np.array([i for i in range(sz)])
        rads = np.array([x.max_rad() for x in self.x_iters])

        plt.plot(inter_nums, rads)
        plt.show()
