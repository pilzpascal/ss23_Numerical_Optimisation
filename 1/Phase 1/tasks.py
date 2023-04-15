import numpy as np
import sympy as sp


def task_1() -> np.longfloat:
    x = sp.symbols('x')
    f = x**3 - 3*x
    return f


def task_2(inp: np.ndarray) -> np.longfloat:
    inp = (inp - 4)
    return inp * (3 * inp ** 3 + 8 * inp ** 2 + 6 * inp - 24)
