from hashlib import new
from generator import Generator, QuadraticGenerator
from interval_data import IntervalData
from interval_regression import QuadraticIntervalRegression, QIRUndefinedCenter, QIRTol


import numpy as np
import numpy.typing as npt


def work(regression_type: QuadraticIntervalRegression.RegressionType, datas: npt.ArrayLike, additional_information: npt.ArrayLike) -> None:
    for data, info in zip(datas, additional_information):
        regression = QuadraticIntervalRegression.create(regression_type, data)

        params = regression.build_model()
        print(f'{info}: params = {params}')

        regression.plot(True)
        print('###############')

    print('\n')

def data(case: int) -> tuple:
    if case == 0:
        data = QuadraticGenerator().generate(25, 0, 25, [7, 5, -1 / 10])
        em_data = data.add_emissions(5, 5, 1)

    elif case == 1:
        data = QuadraticGenerator().generate(50, 0, 7, [-42, -17, 5])
        em_data = data.add_emissions(5, 5, 1)

    else:
        return None

    return IntervalData(data), IntervalData(em_data)


def main():
    cases = [0, 1]

    for case in cases:
        print(f'Case: {case}')
        idata, em_idata = data(case)

        work(QuadraticIntervalRegression.RegressionType.UndifinedCenter, np.array([idata, em_idata]), np.array(['valid data', 'data with estims']))
        work(QuadraticIntervalRegression.RegressionType.Tol, np.array([idata, em_idata]), np.array(['valid data', 'data with estims']))

    return


if __name__ == '__main__':
    main()
