from __future__ import annotations
from code import interact

from generator import Generator, Data
from interval import Interval

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt


class IntervalData(Data):
    def __init__(self, data: Generator.DataInfo) -> None:
        super().__init__()
        self._factors: npt.ArrayLike = data.factors()
        
        rng = np.random.default_rng()
        self._responses: npt.ArrayLike = np.array([
            Interval(response - np.abs(rng.normal(0, 5)), response + np.abs(rng.normal(0, 5))) for response in data.responses()
        ])

    def plot(self, show: bool = True) -> None:
        for x, y in zip(self._factors, self._responses):
            a, b  = y.interval_boundaries()
            plt.plot([x, x], [a, b], 'k')

        if show:
            plt.show()

    def add_emissions(self, emissions_num: int, loc: float = 0, scale: float = 1) -> Data:
        new_data = self.copy()

        for _ in range(emissions_num):
            ind = self._rng.integers(0, self.size())

            new_data._responses[ind].add_number(self._rng.normal(loc, scale))

        return new_data

    def copy(self) -> Data:
        return None
        
