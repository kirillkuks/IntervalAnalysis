from __future__ import annotations

from numpy.core import function_base
from numpy.core.numeric import tensordot

from interval_matrix import DefaultIntervalMatrix, Interval, IntervalVector, IntervalMatrix, Matrix

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

class Task:
    dim = 2

    @staticmethod
    def f1(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Task.dim

        return ivector.at(0).scale(2) + ivector.at(1).scale(3) - Interval(4, 5)

    @staticmethod
    def f2(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Task.dim

        return ivector.at(0) / ivector.at(1) - Interval(1, 2)

    @staticmethod
    def J11(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Task.dim

        return Interval(2, 2)

    @staticmethod
    def J12(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Task.dim

        return Interval(3, 3)

    @staticmethod
    def J21(ivector: IntervalVector) -> Interval:
         assert ivector.sz() == Task.dim

         return Interval(1, 1) / ivector.at(1)

    @staticmethod
    def J22(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Task.dim

        return ( ivector.at(0) / (ivector.at(1) ** 2) ).scale(-1)

    @staticmethod
    def f1l(x: float) -> float:
        return (4 - 2 * x) / 3

    @staticmethod
    def f1r(x: float) -> float:
        return (5 - 2 * x) / 3

    @staticmethod
    def f2l(x: float) -> float:
        return x

    @staticmethod
    def f2r(x: float) -> float:
        return x / 2


class Test:
    dim = 2

    @staticmethod
    def f1(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return ivector.at(0) ** 2 + ivector.at(1) ** 2 - Interval(1, 1)

    @staticmethod
    def f2(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return ivector.at(0) - ivector.at(1) ** 2

    @staticmethod
    def J11(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return ivector.at(0).scale(2)

    @staticmethod
    def J12(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return ivector.at(1).scale(2)

    @staticmethod
    def J21(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return Interval(1, 1)

    @staticmethod
    def J22(ivector: IntervalVector) -> Interval:
        assert ivector.sz() == Test.dim

        return ivector.at(1).scale(-2)

class JacobianInterval:
    @staticmethod
    def create(functions: npt.ArrayLike) -> JacobianInterval:
        size = functions.size // 2
        assert size > 0
        assert functions[0].size == size

        return JacobianInterval(functions)

    def __init__(self, functions: npt.ArrayLike) -> None:
        self.size = functions.size // 2
        self.jacobian = functions

    def sz(self) -> int:
        return self.size

    def interval_matrix(self, ivector: IntervalVector) -> IntervalMatrix:
        assert self.sz() == ivector.sz()

        imatrix = DefaultIntervalMatrix(self.size)
        for i in range(self.size):
            for j in range(self.size):
                imatrix.set(i, j, self.jacobian[i][j](ivector))
            
        return imatrix


class NonlinearIntervalSole:
    @staticmethod
    def create(functions: npt.ArrayLike, jacobian: JacobianInterval, x: IntervalVector) -> NonlinearIntervalSole:
        size = functions.size

        assert size == x.sz()
        assert size == jacobian.sz()

        return NonlinearIntervalSole(functions, jacobian, x)

    def __init__(self, functions: npt.ArrayLike, jacobian: JacobianInterval, x: IntervalVector) -> None:
        self.F = functions
        self.J = jacobian
        self.X = x

        self.x_iters = []
        self.max_ites = 500

    def functions_at(self, ivector: IntervalVector) -> IntervalVector:
        size = ivector.sz()
        assert size == self.F.size

        return IntervalVector.create(np.array([
            self.F[i](ivector) for i in range(size)
        ]))

    def solve(self, eps = 1e-5) -> None:
        x_k = self.X
        counter = 0

        self.x_iters.append(x_k)

        while counter < self.max_ites:
            x_ = x_k.center_point()
            xi_ = IntervalVector.create(np.array([
                Interval(x, x) for x in x_
            ]))

            F_x = self.functions_at(xi_)

            imatrix = self.J.interval_matrix(x_k)
            Lambda = imatrix.mid_matrix().inverse()

            C = Matrix.identity(x_k.sz()).interval_matrix().sub_interval_matrix(imatrix.mul_matrix(Lambda))

            LambdaF_x = Lambda.mul_interval_vector(F_x)
            X_x_ = x_k - xi_

            new_x = xi_ - LambdaF_x - C.mul_vector(X_x_)

            x_k = IntervalVector.intersection(x_k, new_x)
            # print('X_1')
            # print(x_k.at(0).interval_boundaries())
            # print(x_k.at(1).interval_boundaries())

            counter += 1
            self.x_iters.append(x_k)

            if IntervalVector.diff(self.x_iters[counter], self.x_iters[counter - 1]) < eps:
                break

        print('End')
        print(f'Iters num = {counter}')

        print('Result')
        print(self.x_iters[-1].at(0).interval_boundaries())
        print(self.x_iters[-1].at(1).interval_boundaries())
        
        self._plot_interval_2Dbars()
        self._plot_bars_radiuses()
        self._plot_convergence()

    def _plot_F(self, a: float, b: float) -> None:
        x = np.linspace(a, b, 100);
        ys = np.array([
            Task.f1l(x), Task.f1r(x), Task.f2l(x), Task.f2r(x)
        ])
        
        for y in ys:
            plt.plot(x, y, 'k')

    def _plot_interval_2Dbars(self) -> None:
        for x_k, i in zip(self.x_iters, range(len(self.x_iters))):
            a, b = x_k.at(0).interval_boundaries()
            c, d = x_k.at(1).interval_boundaries()
            plt.plot(np.array([a, b, b, a, a]), np.array([c, c, d, d, c]), label=f'iter num = {i}')
        
        # plt.legend()
        plt.title('Boxes')
        self._plot_F(0, 5)
        plt.savefig('NonlinearBoxes.png')
        plt.show()

    def _plot_bars_radiuses(self) -> None:
        sz = len(self.x_iters)

        inter_nums = np.array([i for i in range(sz)])
        rads = np.array([x.max_rad() for x in self.x_iters])

        plt.semilogy(inter_nums, rads)
        plt.title('Radiuses')
        plt.savefig('NonlinearRads.png')
        plt.show()

    def _plot_convergence(self) -> None:
        sz = len(self.x_iters)

        last_box = self.x_iters[-1]

        iter_nums = np.array([i for i in range(sz)])
        boxes_dist = np.array([IntervalVector.diff(iv, last_box) for iv in self.x_iters])

        # center_deltas = np.array([np.linalg.norm(x.center_point() - last_center) for x in self.x_iters])

        plt.semilogy(iter_nums, boxes_dist)
        plt.title('Convergence')
        plt.savefig('NonlinearConv.png')
        plt.show()
