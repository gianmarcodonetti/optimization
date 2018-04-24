import numpy as np


def sen(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def sin_amplitude_obj(x, gain=5):
    return gain - np.abs(x / 20) + sen(31 / 2 * x) + cos(1 * x - 1) + sen(7 / 2 * x + 2)


def sin_amplitude_basic_obj(x, gain=5):
    return gain - np.abs(x / 2) + sen(x)


def neighbour_sin_amplitude(h, k=10, e=10):
    return h + k * np.log((1 + e) ** 0.5) * np.random.normal()
