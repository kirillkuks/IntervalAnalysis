from __future__ import annotations
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt

import numpy as np
import numpy.typing as npt


class Data(ABC):
    def __init__(self) -> None:
        self._factors: npt.ArrayLike
        self._responses: npt.ArrayLike
        self._rng = np.random.default_rng()

    @abstractmethod
    def plot(self, show: bool = True) -> None:
        pass

    @abstractmethod
    def add_emissions(self, emissions_num: int, loc: float = 0, scale: float = 1) -> Data:
        pass

    @abstractmethod
    def copy(self) -> Data:
        pass

    def factors(self) -> npt.ArrayLike:
        return self._factors

    def responses(self) -> npt.ArrayLike:
        return self._responses

    def size(self) -> int:
        return len(self._factors)


class Generator(ABC):
    class DataInfo(Data):
        def __init__(self, factors: npt.ArrayLike, responses: npt.ArrayLike) -> None:
            assert len(factors) == len(responses)
            super().__init__()

            self._factors: npt.ArrayLike = np.copy(factors)
            self._responses: npt.ArrayLike = np.copy(responses)

        def plot(self, show: bool = True) -> None:
            plt.plot(self._factors, self._responses, 'o')

            if show:
                plt.show()

        def add_emissions(self, emissions_num: int, loc: float = 0, scale: float = 1) -> Data:
            new_data = self.copy()

            for _ in range(emissions_num):
                ind = self._rng.integers(0, self.size())

                new_data._responses[ind] += self._rng.normal(loc, scale)

            return new_data

        def copy(self) -> Data:
            return Generator.DataInfo(np.copy(self._factors), np.copy(self._responses))

    def __init__(self) -> None:
        pass

    @abstractmethod
    def generate(self, size: int, a: float, b: float, model_params: npt.ArrayLike) -> DataInfo:
        pass


class QuadraticGenerator(Generator):
    dim: int = 1
    params_num: int = 3
    default_params: npt.ArrayLike = np.array([0, 0, 1])

    @staticmethod
    def model(x: float, params: npt.ArrayLike = default_params) -> float:
        assert len(params) == QuadraticGenerator.params_num

        return params[0] + x * params[1] + x * x * params[2]

    def __init__(self) -> None:
        super().__init__()

    def generate(self, size: int, a: float, b: float, model_params: npt.ArrayLike = default_params) -> Generator.DataInfo:
        delta = (b - a) / size

        return Generator.DataInfo(
            np.array([a + delta * i for i in range(size)]),
            np.array([self.__class__.model(a + delta * i, model_params) for i in range(size)])
        )
