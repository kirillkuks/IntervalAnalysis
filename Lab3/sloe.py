from __future__ import annotations

from interval import Interval
from interval_vector import IntervalVector
from interval_matrix import IntervalMatrix

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from scipy.optimize import dual_annealing, shgo


class Line:
    @staticmethod
    def create(a: float, b: float, c: float) -> Line:
        assert a != 0 or b != 0
        return Line(a, b, c)

    def __init__(self, a: float, b: float, c: float) -> None:
        self.a: float = a
        self.b: float = b
        self.c: float = c

    def func(self, x: float) -> float:
        if self.b != 0:
            return -self.a / self.b * x + self.c / self.b
        
        else:
            return self.c / self.a

    def plot(self, a: float, b: float, show: bool = True) -> None:
        plt.plot([a, b], [self.func(a), self.func(b)])

        if show:
            plt.show()


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
        res = shgo(lambda x: -self.tol(x), bounds=list(zip(lw, up)))

        return np.array([x for x in res.x])

    def max_tol_v(self, v: npt.ArrayLike) -> npt.ArrayLike:
        lw = [-10] * self.columns_num
        up = [10] * self.columns_num
        res = shgo(lambda x: -self.tol_v(x, v), bounds=list(zip(lw, up)))

        return np.array([x for x in res.x])

    def plot_mid_conds(self) -> None:
        a = 0
        b = 5

        lines = []

        for i in range(self.b.sz()):
            a = self.A.at(i, 0).center()
            b = self.A.at(i, 1).center()

            c = self.b.at(i).center()

            lines.append(Line.create(a, b, c))

        for line in lines:
            line.plot(a, b, False)
        
        plt.show()

    def calc_system_vals(self, x: npt.ArrayLike) -> None:
        assert x.size == self.columns_num

        for i in range(self.lines_num):
            interval = Interval(0, 0)

            for j in range(self.columns_num):
                interval = interval + self.A.at(i, j).scale(x[j])
                # print(f'!! {x[j]}, {self.A.at(i, j).interval_boundaries()}, {self.A.at(i, j).scale(x[j]).interval_boundaries()}, {interval.interval_boundaries()}')
            
            print(f'line: {i}, val: {interval.interval_boundaries()}')

    def change_right_vector(self) -> Sloe:
        # v = np.array([self.b.at(i).abs() + 0.01 for i in range(self.lines_num)])
        # print(f'v: {v}')
        v = np.array([1.0 for _ in range(self.lines_num)])
        arg_max = self.max_tol_v(v)
        tol_v = self.tol_v(arg_max, v)

        print(arg_max)
        print(tol_v)

        for i in range(self.b.sz()):
            print(f'{i}: [{self.b.at(i).interval_boundaries()}]')

        delta_b = IntervalVector.create(np.array([Interval(-1, 1).scale(v[i] * (np.abs(tol_v))) for i in range(self.lines_num)]))
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
        for i in range(self.lines_num):
            for j in range(self.columns_num):
                print(*self.A.at(i, j).interval_boundaries(), end=' | ')
            
            print()

        new_A = self.A

        arg_max = self.max_tol()
        max_tol = self.tol(arg_max)

        new_sloe: Sloe = None

        print(new_A.mid())
        print(np.linalg.cond(new_A.mid()))
        
        max_args = [arg_max]

        counter = 0
        max_counter = 25

        while max_tol < -1e-11:
            if counter == max_counter:
                break

            counter += 1

            new_A = self._narrow_matrix(new_A)
            new_sloe = Sloe(new_A, self.b)

            arg_max = new_sloe.max_tol()
            max_tol = new_sloe.tol(arg_max)

            max_args.append(arg_max)
        
            print(arg_max)
            print(max_tol)

        print('Narrow matrix')

        for i in range(self.lines_num):
            for j in range(self.columns_num):
                print(*new_A.at(i, j).interval_boundaries(), end=' | ')
            
            print()

        print('Max argss')
        print(*self._reshape(max_args))
        x, y = self._reshape(max_args)
        for x_k, y_k, i in zip(x, y, range(len(x))):
            plt.plot(x_k, y_k, 'o', label=f'iter: {i}')
        plt.title('Matrix correction')
        plt.savefig('images/MatrixCorrection.png')
        plt.show()

        return new_sloe

        # taus = self.max_tol()
        # delta = min(np.array([
        #     np.sum([np.abs(taus[j]) * self.A.at(i, j).rad() for j in range(self.columns_num)])
        # for i in range(self.lines_num)]))

        # print(delta)
        # E = self._correction_matrix(delta, taus)

        # for i in range(self.lines_num):
        #     for j in range(self.columns_num):
        #         print(*E.at(i, j).interval_boundaries(), end=' | ')
            
        #     print()

        # print('A')
        # for i in range(self.lines_num):
        #     for j in range(self.columns_num):
        #         print(*self.A.at(i, j).interval_boundaries(), end=' | ')
            
        #     print()

        # new_A = self.A.kaucher_sub(E)
        # print('New A')

        # for i in range(self.lines_num):
        #     for j in range(self.columns_num):
        #         print(*new_A.at(i, j).interval_boundaries(), end=' | ')
            
        #     print()

        # new_sloe = Sloe(new_A, self.b)
        # arg_max = new_sloe.max_tol()
        # print(new_sloe.tol(arg_max))

        # return new_sloe

    def change_matrix_by_line(self, line_num: int) -> Sloe:
        for i in range(self.lines_num):
            for j in range(self.columns_num):
                print(*self.A.at(i, j).interval_boundaries(), end=' | ')
            
            print()

        new_A = self.A

        arg_max = self.max_tol()
        max_tol = self.tol(arg_max)

        new_sloe: Sloe = None

        print(new_A.mid())
        print(np.linalg.cond(new_A.mid()))
        
        max_args = [arg_max]

        counter = 0
        max_counter = 25

        while max_tol < -1e-11:
            if counter == max_counter:
                break

            counter += 1

            new_A = self._narrow_matrix_by_line(new_A, line_num)
            new_sloe = Sloe(new_A, self.b)

            arg_max = new_sloe.max_tol()
            max_tol = new_sloe.tol(arg_max)

            max_args.append(arg_max)
        
            print(arg_max)
            print(max_tol)

        print('Narrow matrix')

        for i in range(self.lines_num):
            for j in range(self.columns_num):
                print(*new_A.at(i, j).interval_boundaries(), end=' | ')
            
            print()

        print('Max argss')
        print(*self._reshape(max_args))
        x, y = self._reshape(max_args)
        for x_k, y_k, i in zip(x, y, range(len(x))):
            plt.plot(x_k, y_k, 'o', label=f'iter: {i}')
        plt.title('Matrix correction by line')
        plt.savefig(f'images/MatrixCorrectionByLine{line_num}.png')
        plt.show()

        return new_sloe

    def rve(self) -> float:
        cond = np.linalg.cond(self.A.mid())
        arg_max = self.max_tol()

        return cond * self.tol(arg_max)

    def ive(self) -> float:
        cond = np.linalg.cond(self.A.mid())
        arg_max = self.max_tol()

        return cond * np.sqrt(np.sum(np.array([elem ** 2 for elem in arg_max]))) * self.tol(arg_max) / self.b.norm()

    def _reshape(self, points: npt.ArrayLike) -> tuple:
        assert len(points) > 0
        x, y = [], []

        for point in points:
            assert len(point) == 2

            x.append(point[0])
            y.append(point[1])

        return np.array(x), np.array(y)

    def _narrow_matrix(self, imatrix: IntervalMatrix) -> IntervalMatrix:
        lines, columns = imatrix.sz()

        return IntervalMatrix.create(
            np.array([
                IntervalVector.create(np.array([
                    imatrix.at(i, j).narrow()
                for j in range(columns)]))
            for i in range(lines)])
        )

    def _narrow_matrix_by_line(self, imatrix: IntervalMatrix, line_num: int) -> IntervalMatrix:
        lines, columns = imatrix.sz()
        assert 0 <= line_num < lines

        return IntervalMatrix.create(
            np.array([
                IntervalVector.create(np.array([
                    imatrix.at(i, j).narrow() if i == line_num else imatrix.at(i, j)
                for j in range(columns)]))
            for i in  range(lines)])
        )

    def _correction_matrix(self, K: float, taus: npt.ArrayLike) -> IntervalMatrix:
        assert K > 0

        IntervalVector.create(np.array([
                    Interval.create_mirror_interval(K / taus[j] / self.columns_num)
                for j in range(self.columns_num)]))

        return IntervalMatrix.create(
            np.array([
                IntervalVector.create(
                    np.array([
                        Interval.create_mirror_interval(min(K / taus[j] / self.columns_num, self.A.at(i, j).rad()))
                    for j in range(self.columns_num)]))
            for i in range(self.lines_num)])
        )
