from __future__ import annotations


class Interval:
    @staticmethod
    def intersection(i1: Interval, i2: Interval) -> Interval:
        a1, b1 = i1.interval_boundaries()
        a2, b2 = i2.interval_boundaries()

        a, b = max(a1, a2), min(b1, b2)
        assert a <= b

        return Interval(a, b)

    @staticmethod
    def kaucher_sub(i1: Interval, i2: Interval) -> Interval:
        a1, b1 = i1.interval_boundaries()
        a2, b2 = i2.interval_boundaries()

        return Interval(a1 - a2, b1 - b2)

    @staticmethod
    def create_mirror_interval(d: float) -> Interval:
        d = max(-d, d)
        return Interval(-d, d)

    def __init__(self, a: float = 1.0, b: float = 1.0) -> None:
        self._a = a
        self._b = b

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

    def scale(self, mul: int) -> Interval:
        a, b = self._a * mul, self._b * mul

        if mul < 0:
            a, b == b, a
        
        return Interval(a, b)

    def sub_number(self, num: float) -> Interval:
        return Interval(num - self._b, num - self._a)

    def rad(self) -> float:
        return (self._b - self._a) / 2

    def center(self) -> float:
        return (self._a + self._b) / 2

    def abs(self) -> float:
        return max(-self._a, self._b)

    def narrow(self, k: float = 2.0) -> Interval:
        rad = self.rad() / k
        mid = self.center()

        return Interval(mid - rad, mid + rad)
