from __future__ import annotations

from interval import Interval

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt


class Box:
    def __init__(self, intervals: npt.ArrayLike) -> None:
        self.interavls = intervals

    def to_point(self) -> npt.ArrayLike:
        return self.interavls

    def dim(self) -> int:
        return len(self.interavls)

    def copy(self) -> Box:
        new_intervals = []

        for interval in self.interavls:
            new_intervals.append(interval.copy())

        return Box(np.array(new_intervals))

    def at(self, ind: int) -> Interval:
        assert 0 <= ind < len(self.interavls)

        return self.interavls[ind].copy()

    def set(self, ind: int, interval: Interval) -> None:
        assert 0 <= ind < len(self.interavls)

        self.interavls[ind] = interval.copy() 

    def max_rad(self):
        ind = 0
        m_rad = self.interavls[ind].rad()

        for i in range(1, len(self.interavls)):
            rad = self.interavls[i].rad()

            if rad > m_rad:
                m_rad = rad
                ind = i

        return m_rad, ind

    def center(self) -> npt.ArrayLike:
        return np.array([interval.mid() for interval in self.interavls])

    def dist(self, point: npt.ArrayLike) -> float:
        assert len(point) == self.dim()
        
        d = np.array([interval.dist(coord) for interval, coord in zip(self.interavls, point)])
        return np.sqrt(np.sum(np.array([x ** 2 for x in d])))

    def plot_2d_box(self, show: bool = False) -> None:
        assert len(self.interavls) == 2

        i1 = self.at(0)
        i2 = self.at(1)

        plt.plot(np.array([i1.a, i1.b, i1.b, i1.a, i1.a]), np.array([i2.a, i2.a, i2.b, i2.b, i2.a]), 'k')

        if show:
            plt.show()
