from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import numpy.typing as npt

from scipy.optimize import dual_annealing, shgo


class Sloe:
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

    def max_tol(self) -> npt.ArrayLike:
        lw = [-10] * self.columns_num
        up = [10] * self.columns_num
        res = dual_annealing(lambda x: -self.tol(x), bounds=list(zip(lw, up)))

        return np.array([round(x, 4) for x in res.x])
