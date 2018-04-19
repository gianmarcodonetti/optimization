import numpy as np


def cooling(t, alpha=0.99):
    return t * alpha


def neighbour(h, k, E):
    return h + k * np.log((1 + E) ** 0.5) * np.random.normal()


def probability(delta, T):
    return np.e ** (-delta / T)


def simulated_annealing(h, obj_function, T_initial, T_final, ctr_max, alpha=0.99, k=10, E=10, minimization=True,
                        verbose=False):
    if minimization:
        improvement = lambda x: x <= 0
    else:
        improvement = lambda x: x >= 0
    T = T_initial
    cache = [h]
    while T > T_final:
        if verbose and np.random.random() <= 0.2:
            print("Temperature: {}".format(T))
        for ctr in range(1, ctr_max):
            h_prime = neighbour(h, k, E)
            cache.append(h_prime)
            delta = obj_function(h_prime) - obj_function(h)
            if improvement(delta):
                h = h_prime
            else:
                prob = probability(delta, T)
                if np.random.random() <= prob:
                    h = h_prime
        T = cooling(T, alpha)
    return h, cache
