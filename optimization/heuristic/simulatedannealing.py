from __future__ import print_function
from __future__ import division

import numpy as np
import sys


def cooling(t, alpha=0.99):
    return t * alpha


def simulated_annealing(h, obj_function, neighbour_function, t_initial=80, t_final=1e-100, ctr_max=100, alpha=0.99,
                        minimization=True, verbose=False, caching=True):
    """Simulated Annealing.
    Simulated annealing (SA) is a probabilistic technique for approximating the global optimum of a given function.
    Specifically, it is a metaheuristic to approximate global optimization in a large search space for an optimization
    problem.
    The name and inspiration come from annealing in metallurgy, a technique involving heating and controlled cooling
    of a material to increase the size of its crystals and reduce their defects.

    Args:
        h (float): initial solution
        obj_function (callable): objective (energy) function, to minimize or maximize
        neighbour_function (callable): function able to retrieve a new candidate solution, from a given one
        t_initial (float): initial temperature
        t_final (float): final temperature
        ctr_max (int): number of iterations per cooling process
        alpha (float): temperature decay, should be between 0 and 1
        minimization (bool): whether to minimize or maximize the objective function
        verbose (bool): whether to print details of the iteration process
        caching (bool): whether to store the evaluated solutions

    Returns:
        float, list: the optimal solution together with the cache list
    """

    assert 0 < alpha < 1, "Input param 'alpha' should be in (0, 1), {} is not ok".format(alpha)
    assert 0 < t_final < t_initial, "Input params of temperature should respect:  0 < t_final < t_initial"

    def iteration(_h_current, _neighbour_function, _t, _obj_function, _improvement):
        # 1. Generating new solution
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
            if caching:
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
