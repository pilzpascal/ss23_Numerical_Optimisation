import numpy as np


def hilbert_mat(dim: int) -> np.ndarray:
    Q = np.zeros(shape=(dim, dim))

    for i in range(dim):
        for j in range(dim):
            Q[i, j] = 1 / (i + 1 + j + 1 - 1)

    return Q
