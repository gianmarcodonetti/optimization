import numpy as np
import sys


def cooling(t, alpha=0.99):
    return t * alpha


def neighbour(h, k, E):
    return h + k * np.log((1 + E) ** 0.5) * np.random.normal()


def probability(delta, T):
    return np.e ** (-delta / T)


def simulated_annealing(h, obj_function, T_initial=80, T_final=1e-100, ctr_max=100, alpha=0.99, k=10, E=10,
                        minimization=True, verbose=False):
    """

    Args:
        h (float): initial solution
        obj_function (callable): objective function, to minimize or maximize
        T_initial (float): initial temperature
        T_final (float): final temperature
        ctr_max (int): number of iterations per cooling process
        alpha (float): temperature decay, should be between 0 and 1
        k (float):
        E (float):
        minimization (bool): whether to minimize or maximize the objective function
        verbose (bool): whether to be verbose or not

    Returns:
        float, list:
    """

    assert 0 < alpha < 1, "Input param 'alpha' should be in (0, 1), {} is not ok".format(alpha)
    assert 0 < T_final < T_initial, "Input params of temperature should respect:  0 < T_final < T_initial"

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
