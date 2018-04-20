import numpy as np
import matplotlib.pyplot as plt

from optimization.heuristic import simulatedannealing
from optimization import utils


def basic():
    x = np.linspace(-100, 100, 1000)
    y = utils.sin_amplitude_basic(x)

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Objective Function")
    plt.show()

    h = np.random.randint(10 ** 4, 10 ** 7)
    obj_func = lambda x: - utils.sin_amplitude_basic(x)
    minimization = True

    h_final, cache = simulatedannealing.simulated_annealing(
        h, obj_func, T_initial=80, T_final=1e-100, ctr_max=150, alpha=0.98, k=20, E=20, minimization=minimization,
        verbose=True
    )
    print("Final solution:")
    print("f({}) = {}".format(h_final, -obj_func(h_final)))
    print("Len cache:", len(cache))

    _ = plt.figure(figsize=(14, 10))
    plt.plot([utils.sin_amplitude_basic(c) for c in cache[::20]])
    plt.xlabel('iteration')
    plt.ylabel('value of objective function')
    plt.title("Simulated Annealing searching the optimum")
    plt.show()

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.scatter(h_final, utils.sin_amplitude_basic(h_final), c='r', label='end')
    plt.scatter(h, utils.sin_amplitude_basic(h), c='g', label='start')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Optimal value found")
    plt.legend()
    plt.show()

    return


def advance():
    x = np.linspace(-100, 100, 10000)
    y = utils.sin_amplitude(x)

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Objective Function")
    plt.show()

    h = np.random.randint(10 ** 4, 10 ** 7)
    obj_func = lambda x: - utils.sin_amplitude(x)
    minimization = True

    h_final, cache = simulatedannealing.simulated_annealing(
        h, obj_func, T_initial=80, T_final=1e-100, ctr_max=150, alpha=0.98, k=20, E=20, minimization=minimization,
        verbose=True
    )
    print("Final solution:")
    print("f({}) = {}".format(h_final, -obj_func(h_final)))
    print("Len cache:", len(cache))

    _ = plt.figure(figsize=(14, 10))
    plt.plot([utils.sin_amplitude(c) for c in cache[::20]])
    plt.xlabel('iteration')
    plt.ylabel('value of objective function')
    plt.title("Simulated Annealing searching the optimum")
    plt.show()

    _ = plt.figure(figsize=(14, 10))
    plt.plot(x, y)
    plt.scatter(h_final, utils.sin_amplitude(h_final), c='r', label='end')
    plt.scatter(h, utils.sin_amplitude(h), c='g', label='start')
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
