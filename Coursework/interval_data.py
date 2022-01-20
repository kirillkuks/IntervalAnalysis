from __future__ import annotations
from urllib import response

from generator import Generator, Data
from interval import Interval

import csv

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt


class IntervalData(Data):
    def __init__(self, data: Generator.DataInfo) -> None:
        super().__init__()
        self._data = data.copy()
        self._factors: npt.ArrayLike = data.factors()
        
        rng = np.random.default_rng()
        self._responses: npt.ArrayLike = np.array([
            Interval(response - np.abs(rng.normal(0, 5)), response + np.abs(rng.normal(0, 5))) for response in data.responses()
        ])

    def plot(self, show: bool = True, title: str = None) -> None:
        for x, y in zip(self._factors, self._responses):
            a, b  = y.interval_boundaries()
            plt.plot([x, x], [a, b], 'k')

        if show:
            plt.savefig(f'images/{title}{self.size()}')
            plt.show()

    def add_emissions(self, emissions_num: int, loc: float = 0, scale: float = 1) -> Data:
        new_data = self.copy()

        for _ in range(emissions_num):
            ind = self._rng.integers(0, self.size())

            print(new_data._responses[ind].interval_boundaries())
            new_data._responses[ind] = new_data._responses[ind].add_number(self._rng.normal(loc, scale))
            print(new_data._responses[ind].interval_boundaries())

        return new_data

    def save_as_csv(self, filename: str) -> None:
        filename = filename.replace(' ', '')

        header = ['factors', 'responses']
        data = [[round(factor, 3), response.to_str()] for factor, response in zip(self.factors(), self.responses())]

        with open(f'artifacts/{filename}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header)
            writer.writerows(data)

    def copy(self) -> Data:
        data = IntervalData(self._data)
        
        data._factors = np.copy(self._factors)
        data._responses = np.array([
            response.copy() for response in self._responses
        ])


        return data
        
