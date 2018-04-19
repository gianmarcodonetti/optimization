import numpy as np
import sys


def cooling(t, alpha=0.99):
    return t * alpha


def neighbour(h, k, E):
    return h + k * np.log((1 + E) ** 0.5) * np.random.normal()


def probability(delta, T):
    return np.e ** (-delta / T)


def simulated_annealing(h, obj_function, T_initial, T_final, ctr_max, alpha=0.99, k=10, E=10, minimization=True,
                        verbose=False):
    """

    Args:
        h (float):
        obj_function (callable):
        T_initial (float):
        T_final (float):
        ctr_max (int):
        alpha (float):
        k (float):
        E (float):
        minimization (bool):
        verbose (bool):

    Returns:
        float, list:
    """
    if minimization:
        improvement = lambda x: x <= 0
    else:
        improvement = lambda x: x >= 0
    h_current = h
    T = T_initial
    cache = [h]
    while T > T_final:
        if verbose and np.random.random() < 0.01:
            sys.stdout.write("Temperature: {}\n".format(T))
        for ctr in range(1, ctr_max):
            h_prime = neighbour(h_current, k, E)
            cache.append(h_prime)
            h_current = evaluate_move(h_current, h_prime, T, obj_function, improvement)
        T = cooling(T, alpha)
    return h_current, cache


def evaluate_move(h, h_prime, T, obj_function, improvement):
    delta = obj_function(h_prime) - obj_function(h)
    if improvement(delta):
        h_new = h_prime
    else:
        prob = probability(delta, T)
        if np.random.random() <= prob:
            h_new = h_prime
        else:
            h_new = h
    return h_new
