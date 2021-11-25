from interval_matrix import IntervalMatrix, IntervalVector, Interval, Matrix

import numpy as np
import matplotlib.pyplot as plt


class IntervalSole:
    def __init__(self, A: IntervalMatrix, b: IntervalVector) -> None:
        self.A = A
        self.b = b

        self.L = self._preconditioner_matrix()
        self.x_iters = []

        self.max_iters = 15

    def solve(self) ->IntervalMatrix:
        interval_identity_matrix = Matrix.identity(self.A.sz()).interval_matrix()

        LA = self.A.mul_matrix(self.L)

        m = interval_identity_matrix.sub_interval_matrix(LA)

        Lb = self.L.mul_interval_vector(self.b)

        x_k = self._first_approximation(LA, Lb)
        self.x_iters.append(x_k)

        for _ in range(1, self.max_iters):
            x = Lb + m.mul_vector(x_k)
            x_k = IntervalVector.intersection(x, x_k)

            self.x_iters.append(x_k)

        self._plot_interval_2Dbars()

        print('Result')
        print(x_k.at(0).interval_boundaries())
        print(x_k.at(1).interval_boundaries())

        self._plot_bars_radiuses()
        self._plot_convergence()

    def _preconditioner_matrix(self) -> Matrix:
        return self.A.mid_matrix().inverse()

    def _first_approximation(self, LA: IntervalMatrix, Lb: IntervalVector) -> IntervalVector:
        size = LA.sz()
        assert size == Lb.sz()

        identity_matrix = Matrix.identity(size).interval_matrix()
        C = identity_matrix.sub_interval_matrix(LA)

        eta = C.norm_inf()
        assert(0 <= eta < 1)

        theta = Lb.norm_inf() / (1 - eta)
        x = IntervalVector(size)
        
        for i in range(size):
            x.set(i, Interval(-theta, theta))

        return x

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

    def _plot_convergence(self) -> None:
        sz = len(self.x_iters)

        last_center = self.x_iters[-1].center_point()

        iter_nums = np.array([i for i in range(sz)])
        center_deltas = np.array([np.linalg.norm(x.center_point() - last_center) for x in self.x_iters])

        plt.plot(iter_nums, center_deltas)
        plt.show()
