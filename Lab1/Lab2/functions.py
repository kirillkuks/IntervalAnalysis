from interval import Interval
from box import Box

import numpy as np
import numpy.typing as npt

from abc import ABC, abstractmethod


class Function(ABC):
    def __init__(self, dim) -> None:
        self.dim = dim
        return

    @abstractmethod
    def func(self, X: npt.ArrayLike) -> float:
        pass

    @abstractmethod
    def func_interval(self, intervals: npt.ArrayLike) -> Interval:
        pass

    @abstractmethod
    def search_domain(self) -> Box:
        pass

    @abstractmethod
    def real_min(self) -> npt.ArrayLike:
        pass

    def name(self) -> str:
        return self.__class__.__name__


class ThreeHumpCamelFunction(Function):
    def __init__(self) -> None:
        super().__init__(2)

    def func(self, X: npt.ArrayLike) -> float:
        assert X.size == self.dim
        x, y = X[0], X[1]

        return 2 * (x ** 2) - 1.05 * (x ** 4) + (x ** 6) / 6 + x * y + (y ** 2)

    def func_interval(self, intervals: npt.ArrayLike) -> Interval:
        assert intervals.size == self.dim
        x, y = intervals[0], intervals[1]

        return (x ** 2).scale(2) - (x ** 4).scale(1.05) + (x ** 6).scale(1 / 6) + x * y + (y ** 2)

    def search_domain(self) -> Box:
        return Box(np.array([Interval(-5, 5), Interval(-5, 5)]))

    def real_min(self) -> npt.ArrayLike:
        return np.array([np.array([0, 0])])


class HimmelblauFunction(Function):
    def __init__(self) -> None:
        super().__init__(2)

    def func(self, X: npt.ArrayLike) -> float:
        assert X.size == self.dim
        x, y = X[0], X[1]

        return (x ** 2 + y - 11) ** 2 + (x + y ** 2 - 7) ** 2

    def func_interval(self, intervals: npt.ArrayLike) -> Interval:
        assert intervals.size == self.dim
        x, y = intervals[0], intervals[1]

        return ((x ** 2 + y).add_scalar(-11) ** 2 + (x + y ** 2).add_scalar(-7) ** 2)

    def search_domain(self) -> Box:
        return Box(np.array([Interval(-5, 5), Interval(-5, 5)]))

    def real_min(self) -> npt.ArrayLike:
        return np.array([
            np.array([3, 2]),
            np.array([-2.81, 3.13]),
            np.array([-3.78, -3.28]),
            np.array([3.58, -1.85])
        ])


class SphereFunction(Function):
    def __init__(self) -> None:
        super().__init__(2)

    def func(self, X: npt.ArrayLike) -> float:
        assert X.size == self.dim
        x, y = X[0], X[1]
        
        return x ** 2 + y ** 2

    def func_interval(self, intervals: npt.ArrayLike) -> Interval:
        assert intervals.size == self.dim
        x, y = intervals[0], intervals[1]

        return x ** 2 + y ** 2

    def search_domain(self) -> Box:
        return Box(np.array([Interval(-10, 10), Interval(-10, 10)]))

    def real_min(self) -> npt.ArrayLike:
        return np.array([np.array([0, 0])])


class BoothFunction(Function):
    def __init__(self) -> None:
        super().__init__(2)

    def func(self, X: npt.ArrayLike) -> float:
        assert X.size == self.dim
        x, y = X[0], X[1]

        return (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2

    def func_interval(self, intervals: npt.ArrayLike) -> Interval:
        assert intervals.size == self.dim
        x, y = intervals[0], intervals[1]

        return ((x + y.scale(2)).add_scalar(-7)) ** 2 + ((x.scale(2) + y).add_scalar(-5)) ** 2

    def search_domain(self) -> Box:
        return Box(np.array([Interval(-10, 10), Interval(-10, 10)]))
    
    def real_min(self) -> npt.ArrayLike:
        return np.array([np.array([1, 3])])
