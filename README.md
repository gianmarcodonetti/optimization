# Optimization

Python optimization library.

The developed algorithms are:

- Simulated Annealing
- Reinforced Annealing

## Docs

Documentation and example of usage of the main APIs.

1. [Simulated Annealing](https://github.com/gianmarcodonetti/optimization/blob/master/docs/simulatedannealing.md#Simulated%20Annealing)

## Examples

Containing some examples related to end-to-end usage of the proposed solutions, with different scopes, function to optimize and also framework of reference.


## Heuristic

Classical implementation of heuristic optimization processes.


### Speculative

Speculative implementation of some heuristic optimization processes,
exploiting Software Thread-Level Speculation.

The Software Thread-Level Speculation is the software
simulation of Thread-Level Speculation, and the Thread-Level Speculation
is a parallel technique based on multicore platforms.
The TLS technique specializes in parallelizing irregular programs
with complex dependencies; by heuristically partitioning sequential
program then executing the partitions in a speculatively and parallel
way, a high parallelism could be achieved.

Mainly, the first step is to unroll the iterations in the original
algorithm; the first iteration is the specific non-speculative task,
whereas the others are speculative tasks.
Then, in order to concurrently run all the tasks, speculative tasks
are assigned predicted inputs so that they can run prematurely,
rather than waiting until the results of the first iteration came out.
After the parallel execution finishes, an evaluation measure is held
to check whether the results of the speculative iterations are correct;
the correct ones are kept and on the other hand, those incorrect ones
have to be squashed.
In this way, the total execution time of a sequential implementation
can be shortened.


# References

[1] Zhoukai Wang, Yinliang Zhao, Yang Liu, Cuocuo Lv, (2018) [A speculative parallel simulated annealing algorithm based on Apache Spark](https://www.semanticscholar.org/paper/A-speculative-parallel-simulated-annealing-based-on-Wang-Zhao/e41675b0ddb60b1a1e2b05d8a50d1cbeaeac1c6e) 

[2] Wikipedia, [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing)