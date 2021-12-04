from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

from sloe import Sloe

import numpy as np
import numpy.typing as npt


def init_test():
    line1 = IntervalVector.create(np.array([Interval(0.5, 1.5), Interval(1.5, 2.5)]))
    line2 = IntervalVector.create(np.array([Interval(1.5, 2.5), Interval(-1.5, -0.5)]))
    line3 = IntervalVector.create(np.array([Interval(0.5, 1.5), Interval(-0.5, 0.5)]))

    b = IntervalVector.create(np.array([Interval(3, 5), Interval(-1, 1), Interval(1, 3)]))

    A = IntervalMatrix.create(np.array([line1, line2, line3]))

    return A, b


def init():
    line1 = IntervalVector.create(np.array([Interval(0.5, 1.5), Interval(1.5, 2.5)]))
    line2 = IntervalVector.create(np.array([Interval(1, 1), Interval(-1.5, -0.5)]))
    line3 = IntervalVector.create(np.array([Interval(-0.5, 0.5), Interval(0, 0)]))

    b = IntervalVector.create(np.array([Interval(3, 5), Interval(0, 0), Interval(-1, 1)]))

    A = IntervalMatrix.create(np.array([line1, line2, line3]))

    return A, b   


def main():
    A, b = init_test()

    s = Sloe(A, b)
    print(s.tol(np.array([1, 1])))
    arg_max = s.max_tol()
    print(arg_max)
    print(s.tol(arg_max))

    # print('Change right side')
    # s.change_right_vector()

    print('Change matrix')
    s.change_matrix()

    return


if __name__ == '__main__':
    main()
