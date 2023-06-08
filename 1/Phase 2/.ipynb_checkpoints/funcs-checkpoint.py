import numpy as np
from hilbert_matrix import hilbert_mat


"""
######## Linear CG ########
"""


# minima at [(5., -120., 630., -1120.00000001, 630.)]
def l_f1(x: np.ndarray) -> tuple:
    Q = hilbert_mat(5)
    b = np.ones(shape=5)
    return 0.5 * x.T @ Q @ x - b.T @ x


def l_f1_grad(x: np.ndarray):
    Q = hilbert_mat(5)
    b = np.ones(shape=5)
    return Q @ x - b


# minima at [(-8.00000052e+00,  5.04000025e+02, -7.56000030e+03,  4.62000015e+04,
#        -1.38600004e+05,  2.16216005e+05, -1.68168004e+05,  5.14800010e+04)]
def l_f2(x: np.ndarray) -> tuple:
    Q = hilbert_mat(8)
    b = np.ones(shape=8)
    return 0.5 * x.T @ Q @ x - b.T @ x


def l_f2_grad(x: np.ndarray):
    Q = hilbert_mat(8)
    b = np.ones(shape=8)
    return Q @ x - b


# minima at [(-1.28758021e+01,  1.82700490e+03, -6.35497518e+04,  9.48423728e+05,
#        -7.55539458e+06,  3.58352883e+07, -1.07176657e+08,  2.07239527e+08,
#        -2.58446725e+08,  2.00601048e+08, -8.81030636e+07,  1.67194346e+07)]
def l_f3(x: np.ndarray) -> tuple:
    Q = hilbert_mat(12)
    b = np.ones(shape=12)
    return 0.5 * x.T @ Q @ x - b.T @ x


def l_f3_grad(x: np.ndarray):
    Q = hilbert_mat(12)
    b = np.ones(shape=12)
    return Q @ x - b


# minima at [(-3.06198188e+01,  5.79631108e+03, -2.67656855e+05,  5.30590627e+06,
#        -5.62934738e+07,  3.56180768e+08, -1.41454296e+09,  3.54689896e+09,
#        -5.26881741e+09,  3.31570124e+09,  2.23492076e+09, -4.62144407e+09,
#        -4.79904240e+08,  5.96741325e+09, -4.19120773e+09, -4.68358103e+08,
#         1.17802999e+09,  4.13333400e+08, -7.15516924e+08,  1.98562765e+08)]
def l_f4(x: np.ndarray) -> tuple:
    Q = hilbert_mat(20)
    b = np.ones(shape=20)
    return 0.5 * x.T @ Q @ x - b.T @ x


def l_f4_grad(x: np.ndarray):
    Q = hilbert_mat(20)
    b = np.ones(shape=20)
    return Q @ x - b


# minima at [(-9.82482439e+00,  1.83726227e+03, -8.40808229e+04,  1.63655657e+06,
#        -1.67132563e+07,  9.84342322e+07, -3.42619774e+08,  6.54384034e+08,
#        -3.64246837e+08, -1.23608428e+09,  3.33110919e+09, -4.11395369e+09,
#         3.61640475e+09, -3.04010905e+09,  2.03383571e+09, -6.12820729e+08,
#         5.54105091e+08, -1.97634656e+09,  2.17304903e+09,  2.29862444e+08,
#        -1.61735957e+09, -5.59274698e+08,  1.42743422e+09,  9.15511460e+08,
#        -4.43132977e+08, -2.73444446e+09,  2.35525831e+09,  5.29596440e+08,
#        -1.25096008e+09,  3.87527105e+08)]
def l_f5(x: np.ndarray) -> tuple:
    Q = hilbert_mat(30)
    b = np.ones(shape=30)
    return 0.5 * x.T @ Q @ x - b.T @ x


def l_f5_grad(x: np.ndarray):
    Q = hilbert_mat(30)
    b = np.ones(shape=30)
    return Q @ x - b


"""
######## Nonlinear CG ########
"""


def rosenbrock(x: np.ndarray):
    return 100*(x[1]-x[0]**2)**2 + (1-x[0])**2


def rosenbrock_gradient(x: np.ndarray):
    return np.array([400*x[0]**3 - 400*x[0]*x[1] + 2*x[0] - 2, 200*(x[1]-x[0]**2)])


def nl_f(x: np.ndarray):
    return 150*(x[0]*x[1])**2 + (0.5*x[0] + 2*x[1] - 2)**2


def nl_f_grad(x: np.ndarray):
    return np.array([-2 + 2*x[1] + x[0]*(0.5 + 300*x[1]**2), -8 + 2*x[0] + 8*x[1] + 300*x[0]**2*x[1]])

