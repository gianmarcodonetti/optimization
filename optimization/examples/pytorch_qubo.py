import numpy as np
import torch
import cProfile
import matplotlib.pyplot as plt
from functools import partial

from optimization.heuristic.simulatedannealing import simulated_annealing


def main():
    Q = np.loadtxt('QMtx.dat')
    N = Q.shape[0]

    clipper = lambda t: 0 if t < 0.5 else 1
    vfunc = np.vectorize(clipper)

    def generate_initial_solution(size):
        x_raw = np.random.random(size=size)
        x = vfunc(x_raw)
        return x

    torch.cuda.empty_cache()

    x_torch = torch.cuda.FloatTensor(generate_initial_solution(N))
    Q_torch = torch.cuda.FloatTensor(Q).share_memory_()

    def qubo_obj_function_torch_2(x_torch, Q_torch):
        return torch.matmul(torch.matmul(x_torch, Q_torch), x_torch)

    def neigh_torch(v_torch, v_size, n_bit_to_mutate=2):
        v_copy = v_torch.clone()
        bits = np.random.randint(0, v_size, n_bit_to_mutate)
        for bit in bits:
            v_copy[bit] = 1 - v_copy[bit]
        return v_copy

    h = x_torch

    obj_func_torch = partial(qubo_obj_function_torch_2, Q_torch=Q_torch)
    nb_func_torch = partial(neigh_torch, v_size=N, n_bit_to_mutate=1)

    minimization = True
    t_initial = 10
    t_final = 1e-10
    ctr_max = 100
    alpha = 0.95

    prof = cProfile.Profile()
    args = [h, obj_func_torch, nb_func_torch]
    kwargs = {
        't_initial': t_initial, 't_final': t_final,
        'ctr_max': ctr_max, 'alpha': alpha,
        'minimization': minimization, 'verbose': False, 'caching': True
    }

    h_final, cache = prof.runcall(simulated_annealing, *args, **kwargs)

    prof.print_stats()

    print("Final solution:")
    print("f(x) = {}".format(obj_func_torch(h_final)))
    print("Len cache:", len(cache))
    print("Explored space: {}%".format(len(cache) / 2 ** len(h_final) * 100))

    _ = plt.figure(figsize=(12, 8))
    plt.plot([obj_func_torch(c) for c in cache[::100]])
    plt.xlabel('iteration')
    plt.ylabel('value of objective function')
    plt.show()

    print("Number of 1s: {}".format(h_final.sum()))


if __name__ == '__main__':
    main()
