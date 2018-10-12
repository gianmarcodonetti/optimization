import torch
import numpy as np
from functools import partial

from optimization.heuristic.simulatedannealing import simulated_annealing


def local_annealing(h, solution_size, obj_function, neighbour_function, t_initial=1, t_final=0.9, ctr_max=10,
                    alpha=1e-100, minimization=True, verbose=True, caching=False, break_value=-2050):
    h_local = h
    rev = False
    mts = np.linspace(int(solution_size * 1 / 100), int(solution_size * 10 / 100), num=100)

    if minimization:
        break_condition = lambda x: obj_function(x) <= break_value
    else:
        break_condition = lambda x: obj_function(x) >= break_value

    for i in range(50, 100):
        if verbose:
            print("Iteration {} over 100".format(i))

        divs = int(mts[i])
        nSet = int(solution_size / divs)
        rem = solution_size % divs
        if rev:
            lsDivs = reversed(range(divs + 1))
        else:
            lsDivs = range(divs + 1)

        # Generating start and stop for local shuffling
        for subIdx in lsDivs:
            if subIdx < divs:
                start = subIdx * nSet
                end = (subIdx + 1) * nSet
            else:
                start = subIdx * nSet
                end = subIdx * nSet + rem

            if end - start > 1:
                neigh_shuffling = partial(neighbour_function, start=start, end=end)

                h_local, _ = simulated_annealing(
                    h=h_local, obj_function=obj_function,
                    neighbour_function=neigh_shuffling,
                    t_initial=t_initial, t_final=t_final, ctr_max=ctr_max, alpha=alpha,
                    minimization=minimization, verbose=False, caching=caching
                )

        if break_condition(h_local):
            break

    return h_local


def reinforced_annealing(h, obj_function, neighbour_function, exploitation_function,
                         t_initial=80, t_final=1e-100, ctr_max=100, alpha=0.99,
                         minimization=True, verbose=False, caching=True):
    # 1. Exploration: perform a fast simulated annealing
    h_final, cache = simulated_annealing(h, obj_function, neighbour_function, t_initial, t_final, ctr_max, alpha,
                                         minimization, verbose, caching)

    # 2. Exploitation: perform a local simulated annealing by shuffling
    def neigh_shuffling_torch(v_torch, start, end):
        v_copy = v_torch.clone()
        perm = torch.randperm(end - start).cuda()
        to_cat = []
        if start != 0:
            to_cat.append(v_copy[:start])
        to_cat.append(v_copy[start:end][perm])
        if end != v_torch.shape[0]:
            to_cat.append(v_copy[end:])
        return torch.cat(to_cat)

    h_local = local_annealing(
        h_final, obj_function, exploitation_function,
        t_initial=1, t_final=0.9, ctr_max=10, alpha=1e-100,
        minimization=True, verbose=True, caching=False,
        break_value=-2050
    )
    return h_local
