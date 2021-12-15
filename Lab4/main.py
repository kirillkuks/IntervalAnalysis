from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import numpy.typing as npt

from zyuzin import Zyuzin
from subdifferential_newton import SubdifferentialNewton


def init():
    line1 = IntervalVector.create(np.array([Interval(5, 6), Interval(3, 4)]))
    line2 = IntervalVector.create(np.array([Interval(-1, 1), Interval(2, 3)]))

    b = IntervalVector.create(np.array([Interval(11, 28), Interval(2, 14)]))
    A = IntervalMatrix.create(np.array([line1, line2]))

    return A, b


def init_test():
    line1 = IntervalVector.create(np.array([Interval(2, 4), Interval(-2, 1)]))
    line2 = IntervalVector.create(np.array([Interval(-2, 1), Interval(2, 4)]))

    b = IntervalVector.create(np.array([Interval(-2, 2), Interval(-2, 2)]))
    A = IntervalMatrix.create(np.array([line1, line2]))

    return A, b


def init_subdiff_task1():
    line1 = IntervalVector.create(np.array([Interval(3, 4), Interval(5, 6)]))
    line2 = IntervalVector.create(np.array([Interval(-1, 1), Interval(-3, 1)]))

    b = IntervalVector.create(np.array([Interval(-3, 3), Interval(-1, 2)]))
    A = IntervalMatrix.create(np.array([line1, line2]))

    return A, b


def init_subdiff_task2():
    line1 = IntervalVector.create(np.array([Interval(3, 4), Interval(5, 6)]))
    line2 = IntervalVector.create(np.array([Interval(-1, 1), Interval(-3, 1)]))

    b = IntervalVector.create(np.array([Interval(-3, 4), Interval(-1, 2)]))
    A = IntervalMatrix.create(np.array([line1, line2]))

    return A, b


def main():
    # i1 = Interval(1 / 2, 1 / 4)
    # i2 = Interval(-2 / 3, 2 / 3)

    # print(i1.kaucher_mul(i2).interval_boundaries())
    # # print(i1.kaucher_sub(i2).interval_boundaries())

    # A, b = init()
    # z = Zyuzin.create(A, b)
    # z.solve(1e-16)

    A, b = init_subdiff_task2()
    sn = SubdifferentialNewton(A, b)
    sn.solve(1e-16)

    return


if __name__ == '__main__':
    main()
