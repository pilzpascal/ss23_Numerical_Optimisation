import numpy as np
import sympy as sp


def task_1() -> (sp.Function, sp.Symbol):
    x = sp.symbols('x')
    f = x**3 - 3*x
    return f, x


def task_2() -> (sp.Function, sp.Symbol):
    x = sp.symbols('x')
    f = (x - 9) * (3 * (x - 9) ** 3 + 8 * (x - 9) ** 2 + 6 * (x - 9) - 24)
    return f, x


def task_3() -> (sp.Function, sp.Symbol):
    x = sp.symbols('x')
    f = -sp.exp(-(x - 3)**2)
    return f, x


def task_4() -> (sp.Function, sp.Symbol):
    x = sp.symbols('x')
    f = sp.exp(-(x**2 - (x-1)**4))
    return f, x


def task_5() -> (sp.Function, sp.Symbol):
    x = sp.symbols('x')
    f = sp.exp(-(x-0.6)**4 + (x-0.65)**2)
    return f, x
