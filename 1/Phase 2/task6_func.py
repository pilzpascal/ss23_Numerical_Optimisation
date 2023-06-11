import numpy as np

def function_3() -> (callable, callable, callable):
    """
    Call this function to get the third function, its gradient, and its hessian.
    :return: func, grad, hess
    """
    def func(x: np.ndarray):
        return 0.26 * (x[0] ** 2 + x[1] ** 2) - 0.48 * x[0] * x[1]

    def grad(x: np.ndarray) -> np.ndarray:
        return np.array([0.52*x[0] - 0.48*x[1], -0.48*x[0] + 0.52*x[1]])

    def hess(x: np.ndarray) -> np.ndarray:
        return np.array([[0.52, -0.48],
                         [-0.48, 0.52]])

    return func, grad, hess
