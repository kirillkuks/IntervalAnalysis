from __future__ import annotations

from solver import Solver

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import matplotlib.pyplot as plt


class Zyuzin(Solver):
    @staticmethod
    def create(imatrix: IntervalMatrix, ivector: IntervalVector) -> Zyuzin:
        s1, s2 = imatrix.sz()
        s = ivector.sz()

        assert s1 == s2 == s

        return Zyuzin(imatrix, ivector)

    def __init__(self, imatrix: IntervalMatrix, ivector: IntervalVector) -> None:
        super().__init__()
        self.A = imatrix
        self.b = ivector
        self.size = ivector.sz()

        self.max_iters = 100

    def solve(self, eps: float = 1e-16) -> IntervalVector:
        assert self.A.is_diag_dominant()

        D = self.create_diag()
        E= self.create_non_diag()
        invD = self.inv_diag_matrix(D)

        x_k = IntervalVector.create(np.array([Interval(-10, 10), Interval(-10, 10)]))
        self.x_iters.append(x_k)

        counter = 0
        while counter < self.max_iters:
            counter += 1

            tmp = self.b.kaucher_sub(E.mul_interval_vector(x_k))
            x_k = invD.mul_interval_vector(tmp)

            self.x_iters.append(x_k)
            
            if IntervalVector.diff(self.x_iters[counter], self.x_iters[counter - 1]) < eps:
                break

        print(f'Inters num: {counter}')
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
