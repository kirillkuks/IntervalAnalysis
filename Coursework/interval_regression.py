from __future__ import annotations
from calendar import c

from generator import QuadraticGenerator
from interval import Interval
from interval_data import IntervalData
from interval_matrix import IntervalMatrix
from interval_vector import IntervalVector

from abc import ABC, abstractmethod
from enum import Enum

from scipy.optimize import shgo

import csv

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt


class QuadraticIntervalRegression(ABC):
    class RegressionType(Enum):
        UndifinedCenter = 0,
        Tol = 1

    @staticmethod
    def create(type: RegressionType, data: IntervalData) -> QuadraticIntervalRegression:
        if type == QuadraticIntervalRegression.RegressionType.UndifinedCenter:
            return QIRUndefinedCenter(data)

        if type == QuadraticIntervalRegression.RegressionType.Tol:
            return QIRTol(data)

        return None

    params_num: int = 3

    def __init__(self, data: IntervalData) -> None:
        self.data: IntervalData = data
        self.params: npt.ArrayLike = None
        
        self.name: str = None

    @abstractmethod
    def build_model(self) -> npt.ArrayLike:
        pass

    @abstractmethod
    def additional_plot(self, name: str) -> None:
        pass

    def plot(self, name: str, additional_info: bool = False) -> None:
        if self.params is None:
            self.build_model()

        self.data.plot(False)
        factors = self.data.factors()

        plt.plot(factors, np.array([QuadraticGenerator.model(x, self.params) for x in factors]))

        plt.title(self.name)
        plt.savefig(f'images/{self.name}{name}{self.data.size()}.png')
        plt.show()

        if additional_info:
            self.additional_plot(name)


class QIRUndefinedCenter(QuadraticIntervalRegression):
    def __init__(self, data: IntervalData) -> None:
        super().__init__(data)
        self.name = 'UndefinedCenter'

        self.interval_params: IntervalVector = None

    def build_model(self) -> npt.ArrayLike:
        b0_min, b0_max = -np.inf, np.inf
        b1_min, b1_max = -np.inf, np.inf
        b2_min, b2_max = -np.inf, np.inf

        inds = np.array([0, 1, 2])     
        
        while self._next_comb(inds, self.data.size()):
            b0_h, b0_b = self.b0(inds)
            b0_min, b0_max = max(b0_min, b0_h), min(b0_max, b0_b)

            b1_h, b1_b = self.b1(inds)
            b1_min, b1_max = max(b1_min, b1_h), min(b1_max, b1_b)

            b2_h, b2_b = self.b2(inds)
            b2_min, b2_max = max(b2_min, b2_h), min(b2_max, b2_b)
            
        self.interval_params = IntervalVector.create(np.array([
            Interval(b0_min, b0_max), Interval(b1_min, b1_max), Interval(b2_min, b2_max)
        ]))
        self.params = np.array([self.interval_params.at(i).mid() for i in range(self.interval_params.sz())])

        print(f'b0| min: {b0_min}, max: {b0_max}')
        print(f'b1| min: {b1_min}, max: {b1_max}')
        print(f'b2| min: {b2_min}, max: {b2_max}')

        return np.copy(self.params)

    def additional_plot(self, name: str) -> None:
        self.plot_corridor(name)

    def b0(self, inds: npt.ArrayLike) -> tuple:
        assert len(inds) == self.params_num
        factors = self.data.factors()
        responses = self.data.responses()

        x_i, x_j, x_k = factors[inds[0]], factors[inds[1]], factors[inds[2]]
        y_i, y_j, y_k = responses[inds[0]], responses[inds[1]], responses[inds[2]]

        b_H = y_i.down() if inds[0] == 0 else \
            x_i * x_j * x_k / (x_j - x_i) * (y_i.down() / (x_i * (x_k - x_i)) - y_j.up() / (x_j * (x_k - x_j)) 
            + y_k.down() * (x_j - x_i) / (x_k * (x_k - x_i) * (x_k - x_j)))
        b_B = y_i.up() if inds[0] == 0 else \
            x_i * x_j * x_k / (x_j - x_i) * (y_i.up() / (x_i * (x_k - x_i)) - y_j.down() / (x_j * (x_k - x_j))
            + y_k.up() * (x_j - x_i) / (x_k * (x_k - x_i) * (x_k - x_j)))

        return b_H, b_B

    def b1(self, inds: npt.ArrayLike) -> tuple:
        assert len(inds) == self.params_num
        factors = self.data.factors()
        responses = self.data.responses()

        x_i, x_j, x_k = factors[inds[0]], factors[inds[1]], factors[inds[2]]
        y_i, y_j, y_k = responses[inds[0]], responses[inds[1]], responses[inds[2]]

        b_H = (x_i + x_k) * (x_j + x_k) / (x_j - x_i) * (-y_i.up() / (x_k ** 2 - x_i ** 2) + y_j.down() / (x_k ** 2 - x_j ** 2)
                - y_k.up() * (x_j ** 2 - x_i ** 2) / ((x_k ** 2 - x_j ** 2) * (x_k ** 2 - x_i ** 2)))
        b_B = (x_i + x_k) * (x_j + x_k) / (x_j - x_i) * (-y_i.down() / (x_k ** 2 - x_i ** 2) + y_j.up() / (x_k ** 2 - x_j ** 2)
                - y_k.down() * (x_j ** 2 - x_i ** 2) / ((x_k ** 2 - x_j ** 2) * (x_k ** 2 - x_i ** 2)))

        return b_H, b_B

    def b2(self, inds: npt.ArrayLike) -> tuple:
        assert len(inds) == self.params_num
        factors = self.data.factors()
        responses = self.data.responses()

        x_i, x_j, x_k = factors[inds[0]], factors[inds[1]], factors[inds[2]]
        y_i, y_j, y_k = responses[inds[0]], responses[inds[1]], responses[inds[2]]

        b_H = (y_i.down() / (x_k - x_i) - y_j.up() / (x_k - x_j) + y_k.down() * (x_j - x_i) / ((x_k - x_j) * (x_k - x_i))) / (x_j - x_i)
        b_B = (y_i.up() / (x_k - x_i) - y_j.down() / (x_k - x_j) + y_k.up() * (x_j - x_i) / ((x_k - x_j) * (x_k - x_i))) / (x_j - x_i)

        return b_H, b_B

    def plot_corridor(self, name: str) -> None:
        low_params = np.array([self.interval_params.at(i).down() for i in range(self.interval_params.sz())])
        high_params = np.array([self.interval_params.at(i).up() for i in range(self.interval_params.sz())])

        self.data.plot(False)
        factors = self.data.factors()

        plt.fill_between(factors,
            np.array([QuadraticGenerator.model(x, low_params) for x in factors]),
            np.array([QuadraticGenerator.model(x, high_params) for x in factors])
        )

        plt.savefig(f'images/Corridor{self.name}{name}{self.data.size()}.png')
        plt.show()

    def _next_comb(self, arr: npt.ArrayLike, n: int) -> bool:
        k = len(arr)

        for i in range(k - 1, -1, -1):
            if arr[i] < n - k + i:
                arr[i] += 1

                for j in range(i + 1, k):
                    arr[j] = arr[j - 1] + 1

                return True
        
        return False
                
    def _get_normal_factors(self) -> None:
        factors = self.data.factors()
        shift = -factors[0]

        self.normal_factors = np.array([factor + shift for factor in factors])

    def _max_response_rad(self) -> float:
        responses = self.data.responses()

        return max([response.rad() for response in responses])


class QIRTol(QuadraticIntervalRegression):
    class Tol:
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
            np.sum([self.A.at(i, j).scale(x[j]) for j in range(self.columns_num)]).sub_number(self.b.at(i).mid()).abs()
            for i in range(self.lines_num)])

        def tol_v(self, x: npt.ArrayLike, v: npt.ArrayLike) -> float:
            assert x.size == self.columns_num
            assert v.size == self.lines_num

            return min([1 / v[i] * (self.b.at(i).rad() -
            np.sum([self.A.at(i, j).scale(x[j]) for j in range(self.columns_num)]).sub_number(self.b.at(i).mid()).abs())
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


    def __init__(self, data: IntervalData) -> None:
        super().__init__(data)
        self.name = 'Tol'

        self.A: IntervalMatrix = None
        self.b: IntervalVector = None

        self._build_system()

    def build_model(self) -> npt.ArrayLike:
        tol = QIRTol.Tol(self.A, self.b)
        self.params = tol.max_tol()

        print(f'Tol value: {tol.tol(self.params)}')

        return np.copy(self.params)

    def additional_plot(self, name: str) -> None:
        return None

    def _save_as_csv(self) -> None:
        header = ['factor', 'response']
        data = [[factor, response.to_str()] for factor, response in zip(self.data.factors(), self.data.responses())]

        with open('sample.csv', 'w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(header)


    def _build_system(self) -> None:
        self.A = IntervalMatrix.create(np.array([
            IntervalVector.create(np.array([
                Interval.create_degenerate_interval(1), 
                Interval.create_degenerate_interval(factor), 
                Interval.create_degenerate_interval(factor * factor)
            ])) for factor in self.data.factors()
        ]))

        self.b = IntervalVector.create(np.array([
            response.copy() for response in self.data.responses()
        ]))
