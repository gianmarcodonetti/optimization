import numpy as np


def sen(x):
    return np.sin(x)


def cos(x):
    return np.cos(x)


def sin_amplitude(x, gain=5):
    return gain - np.abs(x / 20) + sen(31 / 2 * x) + cos(1 * x - 1) + sen(7 / 2 * x + 2)


def sin_amplitude_basic(x, gain=5):
    return gain - np.abs(x / 2) + sen(x)


def pipe_functions(functions, zero_value):
    return reduce(lambda res, f: f(res), functions, zero_value)
