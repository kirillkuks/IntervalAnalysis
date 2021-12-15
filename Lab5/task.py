from __future__ import annotations

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt

from interval import Interval


class Task:
    class TaskData:
        @staticmethod
        def create(x: npt.ArrayLike, y: npt.ArrayLike) -> Task.TaskData:
            assert len(x) == len(y)

            return Task.TaskData(x, y)

        def __init__(self, x: npt.ArrayLike, y: npt.ArrayLike) -> None:
            self.x = np.copy(x)
            self.y = np.copy(y)

        def factors(self) -> npt.ArrayLike:
            return np.copy(self.x)

        def responses(self) -> npt.ArrayLike:
            return np.copy(self.y)

        def plot(self, show: bool = True) -> None:
            for x, y in zip(self.x, self.y):
                a, b = y.interval_boundaries()
                plt.plot([x, x], [a, b], 'b')
            
            if show:
                plt.show()


    @staticmethod
    def create(size: int, k: float = 1, b: float = 0) -> Task:
        assert size > 0

        return Task(k, b, size)

    def __init__(self, k: float, b: float, size: int) -> None:
        self.k = k
        self.b = b

        self.size = size

        self.rng = np.random.default_rng()

    def build_task(self) -> TaskData:
        x = np.linspace(0, self.size - 1, self.size)

        y = np.array([
            Interval(
                self.k * (x_k - np.abs(self.rng.normal(0, 5)) + self.b),
                self.k * (x_k + np.abs(self.rng.normal(0, 5)) + self.b)
            ) for x_k in x
        ])

        return Task.TaskData.create(x, y)

    def sz(self) -> int:
        return self.size
