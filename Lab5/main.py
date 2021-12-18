from task import Task
from interval_regression import IntervalRegression

import numpy as np


def main():
    task = Task.create(25)
    # task.build_task().plot()

    iregression = IntervalRegression.create(task)
    iregression.build()
    # iregression.build_point_model()
    # iregression.build_inform_set2D()

    x1 = np.array([5.5 + i for i in range(10)])
    x2 = np.array([50 + i for i in range(10)])
    x3 = np.array([0.0 + i / 2 for i in range(120)])

    iregression.predict(x1)
    iregression.predict(x2)
    iregression.predict(x3)

    return


if __name__ == '__main__':
    main()
