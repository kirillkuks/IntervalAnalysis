from __future__ import annotations

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import numpy.typing as npt

from scipy.optimize import dual_annealing, shgo


class Sloe:
    @staticmethod
    def ident_interval_vector(size: int) -> IntervalVector:
        assert size > 0

        return IntervalVector.create(np.array([Interval(-1, 1) for _ in range(size)]))

    def __init__(self, A: IntervalMatrix, b: IntervalVector) -> None:
        m, n = A.sz()
        assert m == b.sz()

        self.A = A
        self.b = b
        self.lines_num = m
        self.columns_num = n

    def tol(self, x: npt.ArrayLike) -> float:
        assert x.size == self.columns_num

        return min([self.b.at(i).rad() -
        np.sum([self.A.at(i, j).scale(x[j]) for j in range(self.columns_num)]).sub_number(self.b.at(i).center()).abs()
        for i in range(self.lines_num)])

    def tol_v(self, x: npt.ArrayLike, v: npt.ArrayLike) -> float:
        assert x.size == self.columns_num
        assert v.size == self.lines_num

        return min([1 / v[i] * (self.b.at(i).rad() -
        np.sum([self.A.at(i, j).scale(x[j]) for j in range(self.columns_num)]).sub_number(self.b.at(i).center()).abs())
        for i in range(self.lines_num)])

    def max_tol(self) -> npt.ArrayLike:
        lw = [-10] * self.columns_num
        up = [10] * self.columns_num
        res = dual_annealing(lambda x: -self.tol(x), bounds=list(zip(lw, up)))

        return np.array([x for x in res.x])

    def max_tol_v(self, v: npt.ArrayLike) -> npt.ArrayLike:
        lw = [-10] * self.columns_num
        up = [10] * self.columns_num
        res = dual_annealing(lambda x: -self.tol_v(x, v), bounds=list(zip(lw, up)))

        return np.array([x for x in res.x])

    def change_right_vector(self) -> Sloe:
        v = np.array([self.b.at(i).abs() for i in range(self.lines_num)])
        arg_max = self.max_tol_v(v)
        tol_v = self.tol_v(arg_max, v)

        print(arg_max)
        print(tol_v)

        for i in range(self.b.sz()):
            print(f'{i}: [{self.b.at(i).interval_boundaries()}]')

        delta_b = IntervalVector.create(np.array([Interval(-1, 1).scale(v[i] * (np.abs(tol_v) + 0.1)) for i in range(self.lines_num)]))
        for i in range(delta_b.sz()):
            print(f'{i}: [{delta_b.at(i).interval_boundaries()}]')

        new_b = self.b + delta_b
        for i in range(new_b.sz()):
            print(f'{i}: [{new_b.at(i).interval_boundaries()}]')

        sloe = Sloe(self.A, new_b)
        arg_max = sloe.max_tol()
        print(arg_max)
        print(sloe.tol(arg_max))
        return sloe

    def change_matrix(self) -> Sloe:
        arg_max = self.max_tol()
        delta = min(np.array([
            np.sum([np.abs(arg_max[j]) * self.A.at(i, j).rad()   for j in range(self.columns_num)])
        for i in range(self.lines_num)]))

        print(delta)
        return None
