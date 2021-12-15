from task import Task
from interval_regression import IntervalRegression


def main():
    task = Task.create(25)
    task.build_task().plot()

    iregression = IntervalRegression.create(task)
    iregression.build_point_model()

    return


if __name__ == '__main__':
    main()
