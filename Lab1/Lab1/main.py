from interval_matrix import IntervalMatrix, TwoDimmensionalEpsilonMatrix, RegressionMatrix, TomographyMatrix, Matrix
from invertible_matrix_criteria import IncertibleCriterion, BaumanCriterion, RumpCriterion


def find_min_invertible_epsilon(matrix: TwoDimmensionalEpsilonMatrix, criterion: IncertibleCriterion, delta_epsilon: float = 0.01, eps0: float = 0.0) -> float:
    epsilon = eps0
    matrix.set_epsilon(epsilon)

    while criterion.apply(matrix) == True:
        epsilon += delta_epsilon
        matrix.set_epsilon(epsilon)

    return epsilon


def main():
    regression_matrix = RegressionMatrix(0.1)
    tomography_matrix = TomographyMatrix(0.1)

    matricies = [regression_matrix, tomography_matrix]
    criteria = [BaumanCriterion(), RumpCriterion()]

    for matrix in matricies:
        for criterion in criteria:
            eps = find_min_invertible_epsilon(matrix, criterion, 0.0001)
            print(f'{criterion.name()} criterion: eps = {eps} for matrix {matrix.name()}')


if __name__ == '__main__':
    main()
