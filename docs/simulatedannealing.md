# Simulated Annealing

Let me introduce the Simulated Annealing service.

## Setup

Importing the required modules:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np
from functools import partial
import cProfile
import zipfile
import sys

from optimization.heuristic.simulatedannealing import simulated_annealing
```

Set a random seed a the required parameters:

```python
np.random.seed = 123

minimization = True
t_initial = 80
t_final = 1e-80
ctr_max = 100
alpha = 0.5
```

## Reading the data

We are trying to solve a QUBO problem. We have already built our matrix of coefficients and we can read it:

```python
qubo_matrix = np.loadtxt('qubo_matrix.dat')
print("Q shape:", qubo_matrix.shape)

assert qubo_matrix.shape[0] == qubo_matrix.shape[1]

N = qubo_matrix.shape[0]
```

We can now start the annealing process.

## Numpy

First, let's pick a random initial solution and define all the required function for the simulation to start:

```python
def generate_initial_solution(size):
    clipper = lambda t: 0 if t < 0.5 else 1
    vfunc = np.vectorize(clipper)
    x_raw = np.random.random(size=size)
    x = vfunc(x_raw)
    return x

def qubo_obj_function_numpy(x, Q):
    return x.dot(Q).dot(x.T)

def neigh_numpy(v, n_bit_to_mutate=2):
    v_copy = v.copy()
    bits = np.random.randint(0, v.size, n_bit_to_mutate)
    for bit in bits:
        v_copy[bit] = 1 - v_copy[bit]
    return v_copy

obj_func = partial(qubo_obj_function_numpy, Q=Q)
nb_func = partial(neigh_numpy_2, n_bit_to_mutate=2)

h = generate_initial_solution(N)
```

Now, we can start the hunting. We also want to profile the function calls, the we exploit the cProfile service.

```python
prof = cProfile.Profile()
args = [h, obj_func, nb_func]
kwargs = {
    't_initial': t_initial, 't_final': t_final,
    'ctr_max': ctr_max, 'alpha': alpha,
    'minimization': minimization, 'verbose': False, 'caching': True
}

h_final, cache = prof.runcall(simulated_annealing, *args, **kwargs)

prof.print_stats()
```

We should now inspect the final solution and the explorated solution space:

```python
print("Final solution:")
print("f(x) = {}".format(obj_func(h_final)))
print("Len cache:", len(cache))
print("Explored space: {} %".format(len(cache) / 2**len(h_final) * 100))

_ = plt.figure(figsize=(12, 8))
plt.plot([qubo_obj_function_numpy(c, Q) for c in cache[::100]])
plt.xlabel('iteration')
plt.ylabel('value of objective function')
plt.show()

del cache
```


## Pytorch

The ***simulated_annealing*** function I have developed is so generic
that we can also run on GPUs, exploiting, for example,
the **Pytorch** framework.
Indeed, we can define our solutions and the QUBO matrix in
torch Tensors, together with adequate objective and neighbour functions.

Let's recreate the flow we have developed for numpy framework.

Creating the initial solution and the QUBO matrix as torch tensors:
```python
torch.cuda.empty_cache()

h_torch = torch.cuda.FloatTensor(generate_initial_solution(N))
Q_torch = torch.cuda.FloatTensor(Q).share_memory_()
```

Now we can define all the required function for the simulation to start with a *torchy slang*:

```python
def qubo_obj_function_torch(_x_torch, _qubo_torch):
    return torch.matmul(torch.matmul(_x_torch, _qubo_torch), _x_torch)

def neigh_torch(v_torch, v_size, n_bit_to_mutate=2):
    v_copy = v_torch.clone()
    bits = np.random.randint(0, v_size, n_bit_to_mutate)
    for bit in bits:
        v_copy[bit] = 1 - v_copy[bit]
    return v_copy

obj_func_torch = partial(qubo_obj_function_torch, Q_torch=qubo_torch)
nb_func_torch = partial(neigh_torch, v_size=N, n_bit_to_mutat
```


We can exploit the same APIs as before:

```python
prof = cProfile.Profile()
args = [h, obj_func_torch, nb_func_torch]
kwargs = {
    't_initial': t_initial, 't_final': t_final,
    'ctr_max': ctr_max, 'alpha': alpha,
    'minimization': minimization, 'verbose': False, 'caching': True
}

h_final, cache = prof.runcall(simulated_annealing, *args, **kwargs)

prof.print_stats()
```

We should now inspect the final solution and the explorated solution space:

```python
print("Final solution:")
print("f(x) = {}".format(obj_func(h_final)))
print("Len cache:", len(cache))
print("Explored space: {} %".format(len(cache) / 2**len(h_final) * 100))

_ = plt.figure(figsize=(12, 8))
plt.plot([qubo_obj_function_numpy(c, Q) for c in cache[::100]])
plt.xlabel('iteration')
plt.ylabel('value of objective function')
plt.show()

del cache
```
