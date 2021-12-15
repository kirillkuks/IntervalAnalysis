from __future__ import annotations

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from scipy.optimize import linprog 

from interval import Interval
from task import Task


class IntervalRegression:
    @staticmethod
    def create(task: Task) -> IntervalRegression:
        assert task is not None

        return IntervalRegression(task)

    def __init__(self, task: Task) -> None:
        self.task = task
        self.task_data = task.build_task()

    def build(self) -> None:
        pass

    def build_point_model(self) -> None:
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

        print(bw)

        self.task_data.plot(False)
        plt.plot(x, b[0] * x + np.array([b[1] for _ in range(dim)]))
        plt.show()
