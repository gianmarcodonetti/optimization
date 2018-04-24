import numpy as np
import matplotlib.pyplot as plt
from functools import partial

from optimization.examples.objfunctions import sin_amplitude_obj, sin_amplitude_basic_obj, neighbour_sin_amplitude
from optimization.heuristic import simulatedannealing


def basic():
    x = np.linspace(-100, 100, 1000)
    y = sin_amplitude_basic_obj(x)

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Objective Function")
    plt.show()

    h = np.random.randint(10 ** 4, 10 ** 7)
    obj_func = lambda s: - sin_amplitude_basic_obj(s)
    nb_func = partial(neighbour_sin_amplitude, k=10, e=10)
    minimization = True

    h_final, cache = simulatedannealing.simulated_annealing(
        h, obj_func, nb_func, t_initial=80, t_final=1e-100, ctr_max=150, alpha=0.98, minimization=minimization,
        verbose=True
    )
    print("Final solution:")
    print("f({}) = {}".format(h_final, -obj_func(h_final)))
    print("Len cache:", len(cache))

    _ = plt.figure(figsize=(14, 10))
    plt.plot([sin_amplitude_basic_obj(c) for c in cache[::20]])
    plt.xlabel('iteration')
    plt.ylabel('value of objective function')
    plt.title("Simulated Annealing searching the optimum")
    plt.show()

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.scatter(h_final, sin_amplitude_basic_obj(h_final), c='r', label='end')
    plt.scatter(h, sin_amplitude_basic_obj(h), c='g', label='start')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Optimal value found")
    plt.legend()
    plt.show()

    return


def advance():
    x = np.linspace(-100, 100, 10000)
    y = sin_amplitude_obj(x)

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Objective Function")
    plt.show()

    h = np.random.randint(10 ** 4, 10 ** 7)
    obj_func = lambda s: - sin_amplitude_obj(s)
    nb_func = partial(neighbour_sin_amplitude, k=10, e=10)
    minimization = True

    h_final, cache = simulatedannealing.simulated_annealing(
        h, obj_func, nb_func, t_initial=80, t_final=1e-100, ctr_max=150, alpha=0.98, minimization=minimization,
        verbose=True
    )
    print("Final solution:")
    print("f({}) = {}".format(h_final, -obj_func(h_final)))
    print("Len cache:", len(cache))

    _ = plt.figure(figsize=(14, 10))
    plt.plot([sin_amplitude_obj(c) for c in cache[::20]])
    plt.xlabel('iteration')
    plt.ylabel('value of objective function')
    plt.title("Simulated Annealing searching the optimum")
    plt.show()

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.scatter(h_final, sin_amplitude_obj(h_final), c='r', label='end')
    plt.scatter(h, sin_amplitude_obj(h), c='g', label='start')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Optimal value found")
    plt.legend()
    plt.show()

    return


if __name__ == '__main__':
    print("Starting the Simulated Annealing examples...")
    basic()
    advance()
    input("Please, press Enter to end...")
