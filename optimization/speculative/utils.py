from functools import reduce

from optimization.heuristic.simulatedannealing import evaluate_move


def pipe_functions(functions, zero_value):
    return reduce(lambda res, f: f(res), functions, zero_value)


def neighbour_mapper(element, neighbour_function):
    if not element['to_evaluate']:
        h_current = element['state'][0]
        h_prime = neighbour_function(h_current)
        return {'to_evaluate': True, 'state': ((h_current, h_prime), 80), 'path': element['path'], 'to_split': True}
    else:
        element.update({'to_split': False})
        return element


def evaluate_mapper(element, obj_function, improvement=lambda x: x <= 0):
    h, h_prime, T = element['state'][0][0], element['state'][0][1], element['state'][1]
    h_new = evaluate_move(h, h_prime, T, obj_function, improvement)
    take_h_prime = h_new == h_prime
    element.update({'go_to_1': take_h_prime})
    return element


def split_flatmapper(element):
    if element['to_split']:
        return [
            element,
            {'to_evaluate': False, 'state': (element['state'][0][0], element['state'][1]),
             'path': element['path'] + '0'},
            {'to_evaluate': False, 'state': (element['state'][0][1], element['state'][1]),
             'path': element['path'] + '1'}
        ]
    else:
        return [element]
