from functools import partial
from itertools import chain
from copy import deepcopy

from optimization.speculative.utils import split_flatmapper, neighbour_mapper, evaluate_mapper, pipe_functions


def map_iterable(it, func):
    return map(func, it)


def flatten(ll):
    return chain.from_iterable(ll)


def flatmap_iterable(it, func):
    return flatten(map_iterable(it, func))


def create_tree_iterable(initial_list, neighbour_function, tree_depth=10):
    functions = [
        partial(flatmap_iterable, func=split_flatmapper),
        partial(map_iterable, func=partial(neighbour_mapper, neighbour_function=neighbour_function))
    ]
    tree = pipe_functions(functions * (tree_depth), map_iterable(initial_list, neighbour_mapper))
    return map(evaluate_mapper, filter(lambda x: x['to_evaluate'], tree))


def remove_first_move(element):
    element['original_path'] = element['path']
    element['path'] = element['path'][1:]
    return element


def explore_tree_iterable(tree_iterable, tree_depth):
    def iterate(tree):
        go_to_1 = list(filter(lambda x: x['path'] == '', deepcopy(tree)))[0]['go_to_1']

        move = '1' if go_to_1 else '0'
        print(move)

        tree_pruned = map(remove_first_move,
                          filter(lambda x: x['path'][0] == move,
                                 filter(lambda x: len(x['path']) > 0,
                                        deepcopy(tree)
                                        )
                                 )
                          )
        return tree_pruned

    functions = [iterate] * tree_depth
    return pipe_functions(functions, tree_iterable)
