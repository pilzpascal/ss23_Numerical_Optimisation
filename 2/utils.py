import numpy as np
import random


def rnd_vec_l1_norm(n):
    order = random.sample(range(n), n)
    vec = np.zeros(n)
    for idx in order:
        vec[idx] = np.random.uniform(-1+np.sum(abs(vec)), 1-np.sum(abs(vec)))
    return vec
