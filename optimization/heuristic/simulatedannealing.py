import numpy as np
import sys


def cooling(t, alpha=0.99):
    return t * alpha


def simulated_annealing(h, obj_function, neighbour_function, t_initial=80, t_final=1e-100, ctr_max=100, alpha=0.99,
                        minimization=True, verbose=False):
    """

    Args:
        h (float): initial solution
        obj_function (callable): objective function, to minimize or maximize
        neighbour_function (callable): function able to retrieve a neighbour_sin_amplitude solution, from a given one
        t_initial (float): initial temperature
        t_final (float): final temperature
        ctr_max (int): number of iterations per cooling process
        alpha (float): temperature decay, should be between 0 and 1
        minimization (bool): whether to minimize or maximize the objective function
        verbose (bool): whether to be verbose or not

    Returns:
        float, list:
    """

    assert 0 < alpha < 1, "Input param 'alpha' should be in (0, 1), {} is not ok".format(alpha)
    assert 0 < t_final < t_initial, "Input params of temperature should respect:  0 < t_final < t_initial"

    def iteration(_h_current, _neighbour_function, _t, _obj_function, _improvement):
        # 1. generating new solution
        _h_prime = _neighbour_function(_h_current)
        # 2. Making decision
        _h_current = evaluate_move(_h_current, _h_prime, _t, _obj_function, _improvement)
        return _h_current, _h_prime

    if minimization:
        improvement = lambda x: x <= 0
    else:
        improvement = lambda x: x >= 0

    h_current = h
    t = t_initial
    cache = [h]
    while t > t_final:
        if verbose and np.random.random() < 0.01:
            sys.stdout.write("Temperature: {}\n".format(t))
        for ctr in range(1, ctr_max):
            # Do a single iteration
            h_current, h_prime = iteration(h_current, neighbour_function, t, obj_function, improvement)
            cache.append(h_prime)
        t = cooling(t, alpha)
    return h_current, cache


def probability(delta, t):
    return np.e ** (-delta / t)


def evaluate_move(h, h_prime, t, obj_function, improvement):
    if hasattr(h, "persist"):
        print("Input 1 is cached? {}".format(h.is_cached))
        print("Input 2 is cached? {}".format(h_prime.is_cached))
        rdd = True
        h_prime.persist()
        h.persist()
    else:
        rdd = False

    delta = obj_function(h_prime) - obj_function(h)
    if improvement(delta):
        h_new = h_prime
        if rdd:
            h.unpersist()
    else:
        prob = probability(delta, t)
        if np.random.random() <= prob:
            h_new = h_prime
            if rdd:
                h.unpersist()
        else:
            h_new = h
            if rdd:
                h_prime.unpersist()
    return h_new
