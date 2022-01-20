from __future__ import annotations
from ast import In
from turtle import st

import numpy as np


class Interval:
    @staticmethod
    def intersection(i1: Interval, i2: Interval) -> Interval:
        a1, b1 = i1.interval_boundaries()
        a2, b2 = i2.interval_boundaries()

        a, b = max(a1, a2), min(b1, b2)
        assert a <= b

        return Interval(a, b)

    @staticmethod
    def create_degenerate_interval(x: float) -> Interval:
        return Interval(x, x)

    @staticmethod
    def _pos_neg(x: float):
        return max(x, 0), max(-x, 0)

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self._a = a;
        self._b = b;

    def interval_boundaries(self):
        return self._a, self._b

    def copy(self) -> Interval:
        return Interval(self._a, self._b)

    def __add__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(self._a + a, self._b + b)

    def __sub__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(self._a - b, self._b - a)

    def __mul__(self, other: Interval) -> Interval:
        a, b = other.interval_boundaries()

        return Interval(min([self._a * a, self._a * b, self._b * a, self._b * b]),
                        max([self._a * a, self._a * b, self._b * a, self._b * b]))

    def kaucher_mul(self, other: Interval) -> Interval:
        a1, b1 = self.interval_boundaries()
        a2, b2 = other.interval_boundaries()

        a1_pos, a1_neg = self.__class__._pos_neg(a1)
        b1_pos, b1_neg = self.__class__._pos_neg(b1)
        a2_pos, a2_neg = self.__class__._pos_neg(a2)
        b2_pos, b2_neg = self.__class__._pos_neg(b2)

        return Interval(
            max(a1_pos * a2_pos, b1_neg * b2_neg) - max(b1_pos * a2_neg, a1_neg * b2_pos),
            max(b1_pos * b2_pos, a1_neg * a2_neg) - max(a1_pos * b2_neg, b1_neg * a2_pos)
        )

    def kaucher_sub(self, other: Interval) -> Interval:
        return self + other.opp()

    def scale(self, mul: int) -> Interval:
        a, b = self._a * mul, self._b * mul

        if mul < 0:
            a, b == b, a
        
        return Interval(a, b)

    def add_number(self, num: float) -> Interval:
        return Interval(self._a + num, self._b + num)

    def sub_number(self, num: float) -> Interval:
        return Interval(num - self._b, num - self._a)

    def rad(self) -> float:
        return (self._b - self._a) / 2

    def mid(self) -> float:
        return (self._a + self._b) / 2

    def abs(self) -> float:
        return max(-self._a, self._b)

    def dual(self) -> Interval:
        return Interval(self._b, self._a)

    def pro(self) -> Interval:
        if self._a <= self._b:
            return Interval(self._a, self._b)

        return self.dual()

    def mig(self) -> float:
        a, b = self.pro().interval_boundaries()

        if a <= 0 <= b:
            return 0

        return min(np.abs(a), np.abs(b))

    def inv(self) -> Interval:
        a, b = self.interval_boundaries()

        return Interval(1 / a, 1 / b)

    def opp(self) -> Interval:
        a, b =self.interval_boundaries()
        return Interval(-a, -b)

    def down(self) -> float:
        return self._a

    def up(self) -> float:
        return self._b

    def to_str(self) -> str:
        return f'[{self.down()}, {self.up()}]'
