# coding=utf-8
# import copy
# import Individual
import random

import itertools
from greenery.fsm import fsm

# from deap import tools
from algorithm import Layer, generate_random_global_parameter, generate_random_layer_parameter, \
    create_random_valid_layer
import sys

def complete_crossover(ind1, ind2, indpb, config):


    """
    Performs a different crossover:
        External: It involves global parameters and layers as whole elements
        Internal: It involves internal parameters of each layer
    :param config: configuration object
    :param ind1: Individual 1 involved in the crossover
    :param ind2: Individual 2 involved in the crossover
    :param indpb: Prob used into the uniform crossover
    :return: mated individuals
    """

    ind1, ind2 = external_crossover(ind1, ind2, indpb, config)

    ind1, ind2 = internal_crossover(ind1, ind2, indpb)

    return ind1, ind2


def complete_mutation(ind1, indpb, prob_add_remove_layer, config):
    """
    Performs a different mutation:
        External: It involves global parameters and layers as whole elements
        Internal: It involves internal parameters of each layer
    :param n_global_out: global number of outputs
    :param n_global_in: global number of inputs
    :param config: config object
    :param prob_add_remove_layer:
    :param ind1: individual to be mutated
    :param indpb: Prob used into the uniform crossover
    :return: mutated individual
    """

    ind1 = external_mutation(ind1, indpb, config)

    ind1 = internal_mutation_fsm(ind1, indpb, prob_add_remove_layer, config)

    return ind1


def uniform_crossover_dict(dict1, dict2, indpb):
    """
    Method to perform a uniform crossover with common items in two dictionaries
    :param dict1: Dictionary of individual 1
    :param dict2: Dictionary of individual 2
    :param indpb: Cross probability
    :return: mated parameters
    """

    for item in dict1.keys():
        if random.random() < indpb and item in dict2:  # Only crosses common items
            dict1[item], dict2[item] = dict2[item], dict1[item]

    return dict1, dict2


def cut_splice_crossover(layers1, layers2, max_layers, config):
    """
    Method to perform a cut and splice crossover. Designed for layers as whole elements
    :param config: config object
    :param max_layers: Max number of layers
    :param layers1: List of layers of individual 1
    :param layers2: List of layers of individual 2
    :return: mated individuals
    """

    max_layers = config.global_parameters['number_layers']['values'][1]

    cxpoint1 = random.randint(1, len(layers1) - 1)

    # Pick a random layer in the first individual and look for it in the second one

    # Una vez fijado el punto de corte del primer individuo se calculan los posibles puntos de corte en el segundo
    # para que al cruzar los trozos se mantenga una configuracion de capas validas (in/out)
    possible_positions = [p for (p, l) in filter(lambda _p:
                                                 _p[1].type == layers1[cxpoint1].type and
                                                 cxpoint1 + len(layers2) - _p[0] <= max_layers and
                                                 len(layers1) - cxpoint1 + _p[0] <= max_layers,
                                                 list(enumerate(layers2))[1:-1])]   # [1:-1] except first and last layer

    if possible_positions:
        cxpoint2 = random.choice(possible_positions)
        layers1, layers2 = layers1[:cxpoint1] + layers2[cxpoint2:], layers2[:cxpoint2] + layers1[cxpoint1:]

    return layers1, layers2


def external_crossover(indv1, indv2, indpb, config):
    """
    This method crosses global parameters following a uniform crossover and
    layers (as whole elements) following a cut and splice crossover
    :param config: config object
    :param indv1: first individual
    :param indv2: second individual
    :param indpb: crossover probability
    :return: mated individuals
    """
    # Crossover global attributes
    dict1, dict2 = uniform_crossover_dict(indv1.global_attributes.__dict__, indv2.global_attributes.__dict__, indpb)

    indv1.global_attributes.__dict__.update(dict1)
    indv2.global_attributes.__dict__.update(dict2)

    # indv1.global_attributes, indv2.global_attributes = GlobalAttributes(**dict1), GlobalAttributes(**dict2)
    max_layers = config.global_parameters['number_layers']['values'][1]

    # Crossover network layers
    indv1.net_struct, indv2.net_struct = cut_splice_crossover(indv1.net_struct, indv2.net_struct, max_layers, config)

    indv1.global_attributes.number_layers = len(indv1.net_struct)
    indv2.global_attributes.number_layers = len(indv2.net_struct)
    return indv1, indv2


def internal_crossover(ind1, ind2, indpb):
    """
    Method to perform a uniform crossover between each pair of layers
    Both layers are placed in the same position of each set of layers
    If one individual has more layers that the other one, no crossover
    is performed in the first the extra layers
    :param indpb: mutation probability
    :param ind1: individual to be mated
    :param ind2: individual to be mated
    :return: mated individuals
    """
    for layer1, layer2 in zip(enumerate(ind1.net_struct[:-1]), enumerate(ind2.net_struct[:-1])):
        if layer1[1].type == layer2[1].type:
            ind1.net_struct[layer1[0]].parameters, ind2.net_struct[layer2[0]].parameters = uniform_crossover_dict(
                    layer1[1].parameters, layer2[1].parameters, indpb)

    ind1.net_struct[-1].parameters, ind2.net_struct[-1].parameters = uniform_crossover_dict(
        ind1.net_struct[-1].parameters, ind2.net_struct[-1].parameters, indpb
    )

    return ind1, ind2


def external_mutation(ind1, indpb, config):
    """
    Method to perform a mutation over the individual global parameters using a uniform crossover
    :param ind1: individual to be mutated
    :param indpb: mutation probability
    :param config: configuration object
    :return: mutated individual
    """

    for global_parameter in ind1.global_attributes.__dict__.keys():
        if random.random() < indpb:
            ind1.global_attributes.__dict__[global_parameter] = generate_random_global_parameter(global_parameter,
                                                                                                 config)

    return ind1


def internal_mutation_fsm(ind, indpb, new_layer_pb, config):
    num_layers = len(ind.net_struct)
    for pos_layer in range(0, num_layers):

        parameters_dict = ind.net_struct[pos_layer].parameters
        layer_type = ind.net_struct[pos_layer].type
        for parameter in parameters_dict.keys():
            # TODO QUITA LA ÑAPA Alex
            if random.random() < indpb and parameter != 'input_shape' and parameter != 'units':
                parameters_dict[parameter] = generate_random_layer_parameter(parameter, layer_type, config)

    if random.random() < new_layer_pb:
        # Where to insert new layers?
        position = random.choice(range(0, len(ind.net_struct) - 1))
        init_type = ind.net_struct[position].type
        end_type = ind.net_struct[position + 1].type
        the_map = config.fsm['map']
        # Forces the FSM to start at the same type as the selected gen
        the_map['inicial'] = {init_type: init_type}
        state_machine = fsm(alphabet=set(config.fsm['alphabet']),
                            states=set(config.fsm['states']),
                            initial="inicial",
                            finals={end_type},
                            map=the_map)
        # Computes sequences [init_type, ..., end_type] with length between 3 and 5 in order to insert 1 to 3 new layers
        candidates = list(itertools.takewhile(lambda c: len(c) <= 5,
                                              itertools.dropwhile(lambda l: len(l) < 3, state_machine.strings())))

        first_layers = list(set([b[0] for b in candidates]))
        candidates = [random.choice([z for z in candidates if z[0] == first_layers[l]]) for l in
                      range(len(first_layers))]

        # Sizes of the candidates
        sizes = list(set(map(len, candidates)))
        max_layers = config.global_parameters['number_layers']['values'][1]
        sizes = list(filter(lambda s: s <= max_layers - len(ind.net_struct), sizes))
        if sizes:
            # Chooses a size of candidate...
            random_size = random.choice(sizes)
            candidates_size = list(filter(lambda c: len(c) == random_size, candidates))
            if candidates_size:
                # ... and selects it as the layers to be inserted
                candidate = random.choice(candidates_size)
                candidate = list(map(lambda lt: Layer([lt], config), candidate))
                # Updates the individual with new layers
                ind.net_struct = ind.net_struct[:position+1] + candidate[1:-1] + ind.net_struct[position + 1:]

    ind.global_attributes.number_layers = len(ind.net_struct)
    return ind


def internal_mutation_with_restrictions(ind, indpb, new_layer_pb, config):
    """
    Mutates an individual following the input/output restrictions of layers
    :param n_global_out: global number of outputs
    :param n_global_in: global number of inputs
    :param ind: individual to mutate
    :param indpb: probability of mutation
    :param new_layer_pb: probability of adding a new layer to the individual
    :param config:
    :return: mutated individual
    """
    num_layers = len(ind.net_struct)
    for pos_layer in range(0, num_layers):

        parameters_dict = ind.net_struct[pos_layer].parameters
        layer_type = ind.net_struct[pos_layer].type
        for parameter in parameters_dict.keys():
            # TODO QUITA LA ÑAPA Alex, pero se puede dejar por el momento
            if random.random() < indpb and parameter != 'input_shape' and parameter != 'units':
                parameters_dict[parameter] = generate_random_layer_parameter(parameter, layer_type, config)

    if num_layers <= config.global_parameters['number_layers']['values'][1] and random.random() < new_layer_pb:
        # Randomly selects a position
        position = random.randrange(0, len(ind.net_struct) - 1)

        # Randomly selects a layer type depending on the previous layer
        last_layer_type_output = config.layers[ind.net_struct[position].type]['out']
        new_layer = create_random_valid_layer(config, last_layer_type_output)

        # Inserts the new layer right after the selected position
        ind.net_struct.insert(position + 1, new_layer)

    ind.global_attributes.number_layers = len(ind.net_struct)

    return ind
