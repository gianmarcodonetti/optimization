from functools import partial

from optimization.heuristic.speculative.utils import split_flatmapper, neighbour_mapper, evaluate_mapper, pipe_functions


def map_rdd(rdd, func):
    return rdd.map(func)


def flatmap_rdd(rdd, func):
    return rdd.flatMap(func)


def filter_rdd(rdd, func):
    return rdd.filter(func)


def create_tree(initial_rdd, neighbour_function, tree_depth=10):
    functions = [
        partial(flatmap_rdd, func=split_flatmapper),
        partial(map_rdd, partial(neighbour_mapper, neighbour_function=neighbour_function))
    ]
    tree = pipe_functions(functions * tree_depth, initial_rdd.map(neighbour_mapper))
    return tree.filter(lambda x: x['to_evaluate']).map(evaluate_mapper)


def create_tree_iterative(initial_rdd, neighbour_function, tree_depth=10):
    tree = initial_rdd.map(neighbour_mapper)
    while tree_depth > 0:
        tree = tree.flatMap(split_flatmapper).map(lambda x: neighbour_mapper(x, neighbour_function))
        tree_depth -= 1
    return tree.filter(lambda x: x['to_evaluate']).map(evaluate_mapper)


def explore_tree(tree, tree_depth):
    def remove_first_move(_element):
        _element['original_path'] = _element['path']
        _element['path'] = _element['path'][1:]
        return _element

    def iterate(_tree):
        go_to_1 = _tree.filter(lambda x: x['path'] == '').first()['go_to_1']
        move = '1' if go_to_1 else '0'
        tree_pruned = (_tree
                       .filter(lambda x: len(x['path']) > 0)
                       .filter(lambda x: x['path'][0] == move)
                       .map(remove_first_move)
                       )
        return tree_pruned

    functions = [iterate] * tree_depth
    return pipe_functions(functions, tree)
