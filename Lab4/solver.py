from abc import ABC, abstractmethod
from matplotlib import pyplot as plt
import numpy as np

from interval_vector import IntervalVector


class Solver(ABC):
    def __init__(self) -> None:
        self.x_iters = []

    @abstractmethod
    def solve(self, eps: float = 1e-16) -> IntervalVector:
        pass

    def _plot_interval_2Dbars(self, title: str = None, save: bool = True) -> None:
        name = self.__class__.__name__ if title is None else title

        for x_k, i in zip(self.x_iters, range(len(self.x_iters))):
            a, b = x_k.at(0).interval_boundaries()
            c, d = x_k.at(1).interval_boundaries()

            plt.plot(np.array([a, b, b, a, a]), np.array([c, c, d, d, c]), label=f'iter num = {i}')
        
        plt.title(f'{name} boxes')
        plt.legend()
        
        if save:
            plt.savefig(f'{name}Boxes.png')

        plt.show()

    def _plot_bars_radiuses(self, title: str = None, save: bool = True) -> None:
        name = self.__class__.__name__ if title is None else title
        sz = len(self.x_iters)

        inter_nums = np.array([i for i in range(sz)])
        rads = np.array([x.max_rad() for x in self.x_iters])

        plt.title(f'{name} radiuses')
        plt.plot(inter_nums, rads)

        if save:
            plt.savefig(f'{name}Radiuses.png')

        plt.show()

