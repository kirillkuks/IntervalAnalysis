from interval_matrix import IntervalMatrix

from abc import ABC, abstractclassmethod

class IncertibleCriterion(ABC):
    def __init__(self, name: str) -> None:
        self._name = name

    @abstractclassmethod
    def apply(self, matrix: IntervalMatrix) -> bool:
        pass

    def name(self):
        return self._name


class BaumanCriterion(IncertibleCriterion):
    def __init__(self) -> None:
        super().__init__('Bauman')
    
    def apply(self, matrix: IntervalMatrix) -> bool:
        mat = matrix.extreme_matrix_by_index(0)
        det_value = mat.det()

        for i in range(1, matrix.extreme_matrix_num()):
            mat = matrix.extreme_matrix_by_index(i)
            if mat.det() * det_value < 0:
                return False
        
        return True


class RumpCriterion(IncertibleCriterion):
    def __init__(self) -> None:
        super().__init__('Rump')

    def apply(self, matrix: IntervalMatrix) -> bool:
        rad_matrix = matrix.rad_matrix()
        mid_matrix = matrix.mid_matrix()

        s_max = max(rad_matrix.svd())
        s_min = min(mid_matrix.svd())

        return s_max < s_min
