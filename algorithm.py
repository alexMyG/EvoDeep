# coding=utf-8
import random
import sys

from copy import deepcopy

import itertools
from deap import tools
from greenery.fsm import fsm


def dummy_eval(individual):
    return random.random(), random.randrange(2, 10), random.random(), random.random(),


def eval_keras(individual, ke):
    sys.stdout.write(".")
    sys.stdout.flush()

    my_ke = deepcopy(ke)

    metrics_names, scores_training, scores_validation, scores_test, model = my_ke.execute(individual)
    accuracy_training = scores_training[metrics_names.index("acc")]
    accuracy_validation = scores_validation[metrics_names.index("acc")]
    accuracy_test = scores_test[metrics_names.index("acc")]

    number_layers = individual.global_attributes.number_layers

    return accuracy_validation, number_layers, accuracy_training, accuracy_test,


def compare_individuals(ind1, ind2):
    """
    This function is used to check if two individuals object refer to the same individual definition (network structure)
    """
    ind1_string = ind1.toString().replace("\t", "").replace("\n", "").replace(" ", "")
    ind2_string = ind2.toString().replace("\t", "").replace("\n", "").replace(" ", "")

    return ind1_string == ind2_string


def eaMuPlusLambdaModified(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                           stats=None, halloffame=None, verbose=__debug__):
    """This is the :math:`(\mu + \lambda)` evolutionary algorithm.
    :param lambda_:
    :param population: A list of individuals.
    :param toolbox: A :class:`~deap.base.Toolbox` that contains the evolution
                    operators.
    :param mu: The number of individuals to select for the next generation.
    :param lambda_: The number of children to produce at each generation.
    :param cxpb: The probability that an offspring is produced by crossover.
    :param mutpb: The probability that an offspring is produced by mutation.
    :param ngen: The number of generation.
    :param stats: A :class:`~deap.tools.Statistics` object that is updated
                  inplace, optional.
    :param halloffame: A :class:`~deap.tools.HallOfFame` object that will
                       contain the best individuals, optional.
    :param verbose: Whether or not to log the statistics.
    :returns: The final population
    :returns: A class:`~deap.tools.Logbook` with the statistics of the
              evolution.
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = (fit[0], )
        ind.my_fitness = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream

    prev_avg = record["avg"]

    num_generations_no_changes = 0
    print "Size of the population is: " + str(len(population))
    # Begin the generational process
    for gen in range(1, ngen + 1):


        offspring = []
        for _ in xrange(lambda_):
            # Random selection
            ind1, ind2 = map(toolbox.clone, random.sample(population, 2))
            # Crossover
            if random.random() < cxpb:
                ind1, ind2 = toolbox.mate(ind1, ind2)
                del ind1.fitness.values
                del ind2.fitness.values
            # Mutation
            if random.random() < mutpb:
                ind1 = toolbox.mutate(ind1)
                del ind1.fitness.values
            if random.random() < mutpb:
                ind2 = toolbox.mutate(ind2)
                del ind2.fitness.values

            offspring.append(ind1)
            offspring.append(ind2)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = (fit[0], )
            ind.my_fitness = fit

        # Update the hall of fame with the last individuals generated
        if halloffame is not None:
            halloffame.update(offspring)

        if num_generations_no_changes > 5:
            print "MAX GENERATIONS WITH NO CHANGES REACHED. Stopping..."
            record = stats.compile(population) if stats is not None else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
            if verbose:
                print logbook.stream
            return population, logbook

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream

        new_avg = record["avg"]
        same_array = True
        for i in range(0, len(new_avg)):
            # TODO: Parametrizar el error permitido 10e-5
            if abs(prev_avg[i] - new_avg[i]) > 10e-5:
                same_array = False
                num_generations_no_changes = 0
                break

        if same_array:
            num_generations_no_changes += 1
        prev_avg = new_avg

        print "Size of the population is: " + str(len(population))

    return population, logbook


class GlobalAttributes:
    """
    @DynamicAttrs
    """

    def __init__(self, config):
        for global_parameter_name in config.global_parameters.keys():
            setattr(self, global_parameter_name, generate_random_global_parameter(global_parameter_name, config))


class Individual(object):
    def __init__(self, config, n_global_in, n_global_out):

        n_layers_start = 5
        num_min = 3
        self.configuration = config

        self.global_attributes = GlobalAttributes(self.configuration)

        self.net_struct = []


        state_machine = fsm(alphabet=set(config.fsm['alphabet']),
                            states=set(config.fsm['states']),
                            initial="inicial",
                            finals={"Dense"},
                            map=config.fsm['map'])

        candidates = list(itertools.takewhile(lambda c: len(c) <= n_layers_start,
                                              itertools.dropwhile(lambda l: len(l) < num_min,
                                                                  state_machine.strings())))

        first_layers = list(set([b[0] for b in candidates]))
        candidates = [random.choice([z for z in candidates if z[0] == first_layers[l]]) for l in
                      range(len(first_layers))]

        sizes = list(set(map(len, candidates)))
        random_size = random.choice(sizes)
        candidates = filter(lambda c: len(c) == random_size, candidates)

        candidate = random.choice(candidates)
        candidate = map(lambda lt: Layer([lt], config), candidate)
        self.net_struct = candidate
        self.net_struct[0].parameters['input_shape'] = (n_global_in,)
        self.net_struct[-1].parameters['output_dim'] = n_global_out
        self.global_attributes.number_layers = len(self.net_struct)

    def toString(self):

        output = ""
        global_attributes_dictionary = self.global_attributes.__dict__
        for item in sorted(global_attributes_dictionary.keys()):
            output += "Global attribute " + str(item) + ": " + str(global_attributes_dictionary[item]) + "\n"

        output += "Net structure: \n"

        for index, layer in enumerate(self.net_struct):
            output += "\t Layer " + str(index) + "\n"

            output += "\t\t Layer type: " + layer.type + "\n"

            for p in sorted(layer.parameters.keys()):
                output += "\t\t " + p + ": " + str(layer.parameters[p]) + "\n"

        return output

    def __repr__(self):
        return "I: " + ",".join(map(str, self.net_struct))


class Layer:
    """
    Class representing each layer of the Keras workflow
    """

    def __init__(self, possible_layers, config, layer_position=None, n_input_outputs=None):
        """
        Fixed arguments of each layers (those not represented in the individual) such as in or out,
        are direct attributes
        Parameters are under the self.parameters

        :param possible_layers: name of possible next layers
        :param config: configuration object
        :param layer_position: position of the layer to be added

        """

        self.type = random.choice(possible_layers)
        self.parameters = {}

        for param in config.layers[self.type].keys():

            if param != "parameters":
                setattr(self, param, config.layers[self.type][param])
            else:
                for p in config.layers[self.type][param]:
                    self.parameters[p] = generate_random_layer_parameter(p, self.type, config)

        # Deal with number of neurons in first and last layer
        if layer_position == 'first':
            # self.type = 'Dense'
            self.parameters['input_shape'] = (n_input_outputs,)
        if layer_position == 'last':
            # Last layer is forced to be dense
            self.type = 'Dense'
            self.parameters = dict()
            for param in config.layers[self.type].keys():

                if param != "parameters":
                    setattr(self, param, config.layers[self.type][param])
                else:
                    for p in config.layers[self.type][param]:
                        self.parameters[p] = generate_random_layer_parameter(p, self.type, config)
            self.parameters['output_dim'] = n_input_outputs

    def __repr__(self):
        return "[" + self.type[:2] + "(" + "|".join(map(lambda (k, v): k[:4] + ":" + str(v), self.parameters.items())) + ")]"


def create_random_valid_layer(config, last_layer_output_type, n_input_outputs=None, layer_position=None):
    """
    Generates a new valid randomly generated layer coherent with the previous existent layer
    :param n_input_outputs:
    :param config: configuration object
    :param layer_position: position of the layer to be added
    :param last_layer_output_type: output type of the previous existent layer
    :return:
    """
    possible_layers = []

    for layer_name, layer_config in config.layers.items():
        if layer_config['in'] == last_layer_output_type:
            possible_layers.append(layer_name)
    layer = Layer(possible_layers, config, layer_position, n_input_outputs)

    return layer


def parser_parameter_types(parameter_config, parameter):
    if parameter == "categorical":
        return parameter_config["values"][random.randrange(0, len(parameter_config["values"]))]

    elif parameter == "range":
        return random.randrange(*parameter_config["values"])

    elif parameter == "rangeDouble":
        return round(random.uniform(*parameter_config["values"]), 1)

    elif parameter == "matrixRatio":
        # return gen_matrix_ratio_tuple(parameter_config["aspect_ratio"], n_neurons_prev_layer)
        return parameter_config["aspect_ratio"]

    elif parameter == "categoricalNumeric":
        val = parameter_config["values"][random.randrange(0, len(parameter_config["values"]))]

        if val:
            return val, val
        else:
            return None

    elif parameter == "2Drange":
        return [random.randrange(*parameter_config["values"]) for _ in range(parameter_config["size"])]

    elif parameter == "boolean":
        return bool(random.getrandbits(1))

    else:
        print "PARAMETER " + parameter + " NOT DEFINED"


def generate_random_global_parameter(parameter_name, configuration):
    """
    This method generates a new random value based on
    :param parameter_name: the parameter for which a new value is given
    -parameter_name- can take. This param contains the whole configuration dictionary
    :param configuration:
    :return:
    """
    parameter_type = configuration.global_parameters[parameter_name]["type"]
    parameter_config = configuration.global_parameters[parameter_name]

    return parser_parameter_types(parameter_config, parameter_type)


def generate_random_layer_parameter(parameter_name, layer_type, configuration):
    """
    This method generates a new random value based on
    :param configuration:
    :param layer_type:
    :param parameter_name: the parameter for which a new value is given
    -parameter_name- can take. This param contains the whole configuration dictionary
    :return:
    """
    if "parameters" not in configuration.layers[layer_type]:
        return None

    parameter_type = configuration.layers[layer_type]["parameters"][parameter_name]["type"]
    parameter_config = configuration.layers[layer_type]["parameters"][parameter_name]

    return parser_parameter_types(parameter_config, parameter_type)