from functools import partial

from optimization.speculative.utils import split_flatmapper, neighbour_mapper, evaluate_mapper
from optimization.utils import pipe_functions


def map_rdd(rdd, func):
    return rdd.map(func)


def flatmap_rdd(rdd, func):
    return rdd.flatMap(func)


def filter_rdd(rdd, func):
    return rdd.filter(func)


def create_tree(initial_rdd, tree_depth=10):
    functions = [partial(flatmap_rdd, func=split_flatmapper), partial(map_rdd, func=neighbour_mapper)]
    tree = pipe_functions(functions * (tree_depth), initial_rdd.map(neighbour_mapper))
    return tree.filter(lambda x: x['to_evaluate']).map(evaluate_mapper)


def create_tree_iterative(initial_rdd, tree_depth=10):
    tree = initial_rdd.map(neighbour_mapper)
    while tree_depth > 0:
        tree = tree.flatMap(split_flatmapper).map(neighbour_mapper)
        tree_depth -= 1
    return tree.filter(lambda x: x['to_evaluate']).map(evaluate_mapper)


def explore_tree(tree, tree_depth):
    def remove_first_move(element):
        element['original_path'] = element['path']
        element['path'] = element['path'][1:]
        return element

    def iterate(tree):
        go_to_1 = tree.filter(lambda x: x['path'] == '').first()['go_to_1']
        move = '1' if go_to_1 else '0'
        tree_pruned = (tree
                       .filter(lambda x: len(x['path']) > 0)
                       .filter(lambda x: x['path'][0] == move)
                       .map(remove_first_move)
                       )
        return tree_pruned

    functions = [iterate] * (tree_depth)
    return pipe_functions(functions, tree)
