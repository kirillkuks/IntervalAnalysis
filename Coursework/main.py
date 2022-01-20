from generator import QuadraticGenerator
from interval_data import IntervalData
from interval_regression import QuadraticIntervalRegression


import numpy as np
import numpy.typing as npt


def work(regression_type: QuadraticIntervalRegression.RegressionType, datas: npt.ArrayLike, additional_information: npt.ArrayLike) -> None:
    for data, info in zip(datas, additional_information):
        regression = QuadraticIntervalRegression.create(regression_type, data)

        data.plot(True, info)
        data.save_as_csv(f'{info}{data.size()}')

        params = regression.build_model()
        print(f'{info}: params = {params}')

        regression.plot(info, True)
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

    idata = IntervalData(data) 

    return idata, idata.add_emissions(5, 5, 1)


def main():
    cases = [0, 1]

    for case in cases:
        print(f'Case: {case}')
        idata, em_idata = data(case)

        work(QuadraticIntervalRegression.RegressionType.UndifinedCenter, np.array([idata, em_idata]), np.array(['ValidData', 'DataWithEstims']))
        work(QuadraticIntervalRegression.RegressionType.Tol, np.array([idata, em_idata]), np.array(['ValidData', 'DataWithEstims']))

    return


if __name__ == '__main__':
    main()
