import numpy as np
import random


def rnd_vec_l1_norm(n):
    random.seed(42)
    order = random.sample(range(n), n)
    vec = np.zeros(n)
    for idx in order:
        vec[idx] = np.random.uniform(-1+np.sum(abs(vec)), 1-np.sum(abs(vec)))
    return vec


def transform_problem_to_qp_form(M, y):
    n = M.shape[1]
    # Objective Function parts
    # This is proportional to the true objective function, which works for optimization (See Discord Channel)
    G = M.T @ M
    c = (-M.T @ y)

    G_extended = np.zeros((G.shape[0] * 2, G.shape[0] * 2))
    G_extended[0:G.shape[0], 0:G.shape[1]] = G

    c_extended = np.zeros(c.size * 2)
    c_extended[0:c.shape[0]] = c
    # Constraints

    # Build constraints in the smart way as given in the task assignment hint
    A = np.zeros((2 * n + 1, 2 * n))
    for i in range(0, n):
        A[2 * i, i] = 1
        A[2 * i, n + i] = -1
        A[2 * i + 1, i] = -1
        A[2 * i + 1, n + i] = -1
    A[-1, n:] = 1

    b = np.zeros(2 * n + 1)
    b[-1] = 1
    return G_extended, c_extended, A, b


def objective_function(x, M, y):
    return 0.5 * np.linalg.norm(M.dot(x) - y)**2


def print_matrix(array):
    matrix = ''
    for row in array:
        try:
            for number in row:
                matrix += f'{number}&'
        except TypeError:
            matrix += f'{row}&'
        matrix = matrix[:-1] + r'\\'
    display(Math(r'\begin{bmatrix}'+matrix+r'\end{bmatrix}'))
