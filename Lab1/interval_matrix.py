import numpy as np

from abc import ABC, abstractclassmethod


class Interval:
    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self._a = a;
        self._b = b;

    def interval_boundaries(self):
        return self._a, self._b

    def copy(self):
        return Interval(self._a, self._b)


class Matrix:
    def __init__(self, size: int) -> None:
        self.size = size
        self.matrix = np.array([np.array([0.0 for _ in range(size)]) for _ in range(size)])

    def at(self, i: int, j: int) -> float:
        assert 0 <= i < self.size and 0 <= j < self.size

        return self.matrix[i][j]

    def __setitem__(self, ind: list, value: float) -> None:
        assert len(ind) == 2
        i, j = ind[0], ind[1]
        assert 0 <= i < self.size and 0 <= j < self.size

        self.matrix[i][j] = value

    def __getitem__(self, ind: list) -> float:
        assert len(ind) == 2
        i, j = ind[0], ind[1]
        assert 0 <= i < self.size and 0 <= j < self.size

        return self.matrix[i][j]

    def copy(self):
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                matrix.matrix[i][j] = self.matrix[i][j]

        return matrix

    def det(self) -> float:
        return np.linalg.det(self.matrix)

    def svd(self) -> np.array(float):
        return np.linalg.svd(self.matrix, compute_uv=False)


class IntervalMatrix(ABC):
    def __init__(self, size: int) -> None:
        self.size = size
        self.matrix = np.array([np.array([Interval() for _ in range(size)]) for _ in range(size)])

    @abstractclassmethod
    def at(self, i: int, j: int) -> Interval:
        pass

    def extreme_matrix_num(self) -> int:
        return 2 ** (self.size * self.size)

    def extreme_matrix_by_index(self, ind: int) -> Matrix:
        assert 0 <= ind < self.extreme_matrix_num()

        matrix = Matrix(self.size)
        i, j = 0, 0

        ind_bin = self._binary(ind)

        for digit in ind_bin:
            a, b = self.at(i, j).interval_boundaries()

            if digit == 0:
                matrix[[i, j]] = a
            else:
                matrix[[i, j]] = b

            j += 1
            if j == self.size:
                j = 0
                i += 1

        return matrix

    def _binary(self, num: int):
        digits = [0 for _ in range(self.size * self.size)]
        i = 0

        while num > 0:
            digits[i] = num % 2
            num = num // 2
            i += 1
        
        digits.reverse()
        return np.array(digits)

    def rad_matrix(self) -> Matrix:
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                a, b = self.at(i, j).interval_boundaries()
                matrix[[i, j]] = (b - a) / 2

        return matrix

    def mid_matrix(self) -> Matrix:
        matrix = Matrix(self.size)

        for i in range(self.size):
            for j in range(self.size):
                a, b = self.at(i, j).interval_boundaries()
                matrix[[i, j]] = (a + b) / 2

        return matrix


class TwoDimmensionalEpsilonMatrix(IntervalMatrix):
    def __init__(self, epsilon: float = 0.0, name: str = '') -> None:
        super().__init__(2)
        self.epsilon = epsilon
        self._name = name

    @abstractclassmethod
    def at(self, i: int, j: int) -> Interval:
        pass

    def name(self) -> str:
        return self._name

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = epsilon


class RegressionMatrix(TwoDimmensionalEpsilonMatrix):
    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__(epsilon, 'regression')
        self.matrix[1][0] = Interval(1.1, 1.1)

    def at(self, i: int, j: int) -> Interval:
        assert 0 <= i < self.size and 0 <= j < self.size

        interval = self.matrix[i][j]

        if j == 1:
            return interval.copy()

        a, b = interval.interval_boundaries()
        
        return Interval(a - self.epsilon, b + self.epsilon)


class TomographyMatrix(TwoDimmensionalEpsilonMatrix):
    def __init__(self, epsilon: float = 0.0) -> None:
        super().__init__(epsilon, 'tomography')
        self.matrix[1][0] = Interval(1.1, 1.1)

    def at(self, i: int, j: int) -> Interval:
        assert 0 <= i < self.size and 0 <= j < self.size

        a, b = self.matrix[i][j].interval_boundaries()
    
        return Interval(a - self.epsilon, b + self.epsilon)
