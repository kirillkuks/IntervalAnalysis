from __future__ import annotations
from functions import Function

from interval import Interval
from box import Box

import numpy as np
import numpy.typing as npt

import matplotlib.pyplot as plt


class WorkListNode:
    def __init__(self, func_vaule: float, domain: Box) -> None:
        self.func_value = func_vaule
        self.domain = domain


class WorkList:
    def __init__(self, node: WorkListNode) -> None:
        self.work_list = [node]

    def size(self) -> int:
        return len(self.work_list)

    def at(self, ind: int) -> WorkListNode:
        assert 0 <= ind < self.size()

        return self.work_list[ind]

    def delete_node(self, ind: int) -> None:
        assert 0 <= ind < self.size()

        del self.work_list[ind]

    def add_node(self, node: WorkListNode) -> None:
        ind = 0

        for n in self.work_list:
            if n.func_value > node.func_value:
                break

            ind += 1

        self.work_list.insert(ind, node)

    def all_ests(self) -> None:
        for node in self.work_list:
            print(node.func_value, end=' | ')
        print()


class Globopt:
    def __init__(self, func: Function) -> None:
        self.func = func
        self.x_ks = []

    def globopt0(self, eps: float):
        func = self.func.func_interval
        box = self.func.search_domain()

        X = box.to_point()
        Y = func(X)
        upper_est = Y.max()
        lead_est = upper_est

        work_list = WorkList(WorkListNode(Y.min(), box.copy()))
        self.x_ks.append(box.center())

        counter = 0
        while func(work_list.at(0).domain.to_point()).rad() >= 2 * eps:

        # while counter < 100:
            # print(counter)
            # for i in range(work_list.size()):
            #     for j in range(work_list.at(i).domain.dim()):
            #         inter = work_list.at(i).domain.to_point()[j]
            #         print(inter.a, inter.b, end=' | ')

            #     print()
            # print('#####################################')


            list_size = work_list.size()
            lead_est = upper_est
            lead_ind = 0

            # for i in range(list_size):
            #     p = work_list.at(i).func_value

            #     if p < lead_est:
            #         lead_est = p
            #         lead_ind = i
            
            lead_ind = 0;

            D1 = work_list.at(lead_ind).domain.copy()
            D2 = work_list.at(lead_ind).domain.copy()

            # print('D1')
            # print(D1.at(0).a, D1.at(0).b)
            # print(D1.at(1).a, D1.at(1).b)

            rad, ind = D1.max_rad()

            if rad == 0:
                break

            s = D1.at(ind)
            mid_s = s.mid()

            D1.set(ind, Interval(s.min(), mid_s))
            D2.set(ind, Interval(mid_s, s.max()))

            Y1 = func(D1.to_point())
            Y2 = func(D2.to_point())

            work_list.delete_node(lead_ind)

            work_list.add_node(WorkListNode(Y1.min(), D1.copy()))
            work_list.add_node(WorkListNode(Y2.min(), D2.copy()))

            self.x_ks.append(work_list.at(0).domain.center())
            
            counter += 1

        print(f'Min point = {work_list.at(0).domain.center()}')
        print(f'Min interval = {work_list.at(0).domain.at(0).a} -- {work_list.at(0).domain.at(0).b} || {work_list.at(0).domain.at(1).a} -- {work_list.at(0).domain.at(1).b}')
        print(f'Lead est = {work_list.at(0).func_value}')
        print(f'Counter = {counter}')

        return work_list.at(0).func_value, work_list


    def plot_boxes(self, work_list: WorkList) -> None:
        for i in range(work_list.size()):
            box = work_list.at(i).domain
            box.plot_2d_box()

        plt.title(self.func.name())
        plt.show()

    def plot_boxes_centers(self, work_list: WorkList) -> None:
        x, y = [], []

        for i in range(work_list.size()):
            box = work_list.at(i).domain

            assert box.dim() == 2

            center = box.center()
            x.append(center[0])
            y.append(center[1])

        plt.plot(x, y)
        plt.title(self.func.name())
        plt.show()

    def plot_boxes_rads(self, work_list: WorkList) -> None:
        rads = []
        for i in range(work_list.size()):
            rad, ind = work_list.at(i).domain.max_rad()
            rads.append(rad)

        plt.semilogy(np.array([i for i in range(work_list.size())]), np.array(rads))
        plt.title(self.func.name())
        plt.show()

    def plot_convergence(self, work_list: WorkList) -> None:
        real_min = self.func.real_min()            

        dists = np.array([
            self._min_dist_to_point(real_min, center) for center in self.x_ks
        ])

        plt.semilogy(np.array([i for i in range(len(self.x_ks))]), np.array(dists))
        plt.title(self.func.name())
        plt.show()

    def _min_dist_to_point(self, points: npt.ArrayLike, point: npt.ArrayLike) -> float:
        for p in points:
            assert len(p) == len(point)

        return min([
            np.sqrt(np.sum(np.array([
                (x - y) ** 2 for x, y in zip(p, point)
            ]))) for p in points
        ])
