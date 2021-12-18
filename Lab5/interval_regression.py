from __future__ import annotations
from os import name

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from scipy.optimize import linprog 
from shapely.geometry import Polygon

from interval import Interval
from task import Task


class IntervalRegression:
    @staticmethod
    def create(task: Task) -> IntervalRegression:
        assert task is not None

        return IntervalRegression(task)

    @staticmethod
    def max_diag_point(polygon: Polygon) -> npt.ArrayLike:
        x, y = polygon.exterior.xy
        size = len(x)

        max_dist = -np.inf
        max_i, max_j = -1, -1

        for i in range(size):
            for j in range(i, size):
                dist = np.sqrt((x[i] - x[j]) ** 2 + (y[i] - y[j]) ** 2)

                if max_dist < dist:
                    max_dist = dist
                    max_i, max_j = i, j

        return np.array([(x[max_i] + x[max_j]) / 2, (y[max_i] + y[max_j]) / 2])
        

    @staticmethod
    def mean_point(polygon: Polygon) -> npt.ArrayLike:
        x, y = polygon.exterior.xy
        size = len(x) - 1
        
        mean_x = np.sum(np.array([x_k for x_k in x[:size]]))
        mean_y = np.sum(np.array([y_k for y_k in y[:size]]))

        return np.array([mean_x / size, mean_y / size])

    def __init__(self, task: Task) -> None:
        self.task = task
        self.task_data = task.build_task()

        self.inform_set = None
        
        self.b = np.array([])

        self.image_path = 'images/'

    def predict(self, x: npt.ArrayLike) -> npt.ArrayLike:
        if self.inform_set is None:
            self.inform_set = self.build_inform_set2D(False)

        y = np.array([self._corridor_interval(x_k, self.inform_set) for x_k in x])

        self._plot_corridor(x, y, False)
        self._plot_model(x, False)

        plt.legend()
        plt.savefig(f'{self.image_path}Predict{x[0]}-{x[-1]}.png')
        plt.show()
    
        return y

    def build(self) -> None:
        self.plot_task()

        self.b = self.build_point_model()

        inform_set = self.build_inform_set2D(False)

        plt.plot(self.b[0], self.b[1], 'o', label='точечная регрессия')
        
        b1 = self.__class__.max_diag_point(inform_set)
        plt.plot(b1[0], b1[1], 'o', label='максимальная диагональ')

        b2 = self.__class__.mean_point(inform_set)
        plt.plot(b2[0], b2[1], 'o', label='центр тяжести')

        plt.legend()
        plt.savefig(f'{self.image_path}InformationSet.png')
        plt.show()

        self.plot_builded(
            np.array([self.b, b1, b2]),
            np.array(['точечная регрессия', 'максимальная диагональ', 'центр тяжести'])
        )

        print('Corridor')

        self.plot_corridor()

    def build_inform_set2D(self, show: bool = True) -> Polygon:
        x = self.task_data.factors()
        y = self.task_data.responses()

        p = None

        for factor, response_interval in zip(x, y):
            q = self._plot_strip(factor, response_interval, 0, 5)

            p = q if p is None else p.intersection(q)

        plt.plot(*p.exterior.xy)

        if show:
            plt.show()

        self.inform_set = p
        return p

    def build_point_model(self) -> npt.ArrayLike:
        dim = self.task.sz()

        c = np.array([0.0 for _ in range(2)] + [1.0 for _ in range(dim)])
        bounds = np.array([(None, None) for _ in range(2)] + [(0, None) for _ in range(dim)])

        A_ub = []
        b_ub = []

        x = self.task_data.factors()
        y = self.task_data.responses()

        for i in range(dim):
            a_k = np.array([-x[i], -1] + [0.0 for _ in range(dim)])
            a_k[i + 2] = -y[i].rad()
            A_ub.append(a_k)
            b_ub.append(-y[i].center())

            a_k = np.array([x[i], 1] + [0.0 for _ in range(dim)])
            a_k[i + 2] = -y[i].rad()
            A_ub.append(a_k)
            b_ub.append(y[i].center())

        bw = linprog(method='simplex', c=c, bounds=bounds, A_ub=A_ub, b_ub=b_ub)

        b = bw.x[:2]
        w = bw.x[2:]

        print(b)
        return np.array(b)

    def build_corridor(self, inform_set: Polygon = None, show: bool = True) -> npt.ArrayLike:
        p = inform_set if inform_set is not None else self.build_inform_set2D(False)

        x = self.task_data.factors()
        y = self.task_data.responses()

        corridor_y = []

        for factor in x:
            ix = self._corridor_interval(factor, inform_set)
            corridor_y.append(ix)
            # print(*ix.interval_boundaries())

        self._plot_corridor(x, corridor_y, show)

    def plot_task(self) -> None:
        self.task_data.plot(False)
        self._plot_model(self.task_data.factors(), False)

        plt.legend()

        plt.savefig(f'{self.image_path}Model.png')
        plt.show()

    def plot_builded(self, betas: npt.ArrayLike, names: npt.ArrayLike) -> None:
        self.task_data.plot(False)

        self._plot_model(self.task_data.factors(), False)
        self.plot_builded_model(
            betas,
            names
        )

        plt.legend()
        plt.savefig(f'{self.image_path}BuildedModels.png')
        plt.show()

    def plot_builded_model(self, betas: npt.ArrayLike, names: npt.ArrayLike) -> None:
        assert len(betas) == len(names)

        for beta, name in zip(betas, names):
            x = self.task_data.factors()
            plt.plot(x, self.task.__class__.func(x, beta), label=name)

    def plot_corridor(self) -> None:
        self.task_data.plot(False)
        self.build_corridor(self.inform_set, False)
        self._plot_model(self.task_data.factors(), False)
        plt.savefig(f'{self.image_path}CorridorOfSharedDependencies.png')
        plt.show()     

    def _plot_corridor(self, x: npt.ArrayLike, y: npt.ArrayLike, show: bool = True) -> None:
        assert len(x) == len(y)

        y1, y2 = [], []

        for response in y:
            a, b = response.interval_boundaries()

            y1.append(a)
            y2.append(b)

        plt.fill_between(x, np.array(y1), np.array(y2))

        if show:
            plt.show()

    def _corridor_interval(self, x: float, inform_set: Polygon) -> Interval:
        min_val, max_val = np.inf, -np.inf
        b_x, b_y = inform_set.exterior.xy

        for b1, b0 in zip(b_x, b_y):
            y = self.task.__class__.func(x, [b1, b0])

            min_val = min(y, min_val)
            max_val = max(y, max_val)

        return Interval(min_val, max_val)

    def _plot_strip(self, x: float, y: Interval, a = 0, b = 1) -> Polygon:
        y_a, y_b = y.interval_boundaries()

        p = Polygon([
            (a, y_a - a * x),
            (a, y_b - a * x),
            (b, y_b - b * x),
            (b, y_a - b * x)
        ])

        return p

    def _plot_model(self, x: npt.ArrayLike, show: bool = True) -> None:
        plt.plot(x, self.task.model(x), 'r', label='модель')

        if show:
            plt.show()
