from matplotlib.pyplot import semilogy
from functions import Function, ThreeHumpCamelFunction, HimmelblauFunction, SphereFunction, BoothFunction
from interval import Interval
from box import Box

from globopt0 import Globopt

import numpy as np


def run(function: Function, eps: float) -> None:
    globopt = Globopt(function)
    est, work_list = globopt.globopt0(eps)

    print('Lead boxes')
    for i in range(5):
        print(work_list.at(i).domain.center())
        # print(f'{work_list.at(i).domain.at(0).a} -- {work_list.at(i).domain.at(0).b} || {work_list.at(i).domain.at(1).a} -- {work_list.at(i).domain.at(1).b}')

    globopt.plot_boxes(work_list)
    globopt.plot_boxes_centers(work_list)
    globopt.plot_boxes_rads(work_list)
    globopt.plot_convergence(work_list)


def main():
    # three_hunp_camel = ThreeHumpCamel()
    # est, work_list = globopt0(three_hunp_camel.func_interval, three_hunp_camel.search_domain(), 0.01)

    # plot_boxes(work_list)
    # plot_boxes_centers(work_list)

    # himmelblau = HimmelblauFunction()
    # est, work_list = globopt0(himmelblau.func_interval, himmelblau.search_domain(), 0.01)

    # plot_boxes(work_list)
    # plot_boxes_centers(work_list)

    # sphere = Sphere()
    # est, work_list = globopt0(sphere.func_interval, sphere.search_domain(), 0.01)

    # plot_boxes(work_list)
    # plot_boxes_centers(work_list)

    # run(ThreeHumpCamel(), 0.1)

    run(HimmelblauFunction(), 0.00001)
    # run(BoothFunction(), 0.00001)
    # run(SphereFunction(), 0.0001)
    # run(ThreeHumpCamelFunction(), 0.1)

    return


if __name__ == '__main__':
    main()
