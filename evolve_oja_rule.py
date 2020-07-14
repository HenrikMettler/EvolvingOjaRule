import numpy as np
import matplotlib.pyplot as plt
import functools
import cgp

from learning_rules import *


def f_target(x, w, y):
    oja = y * (x - y * w)
    return oja


def calc_weight_penalty(weights, mode):
    if mode == 1:
        out = np.square(np.sqrt(np.sum(np.square(weights))) - 1)
    elif mode == 0:
        out = np.sqrt(np.sum(np.square(weights)))
    else:
        raise NotImplementedError(
            "Mode for calculating the weight penalty not available, choose 0 or 1"
        )
    return out


def objective(individual, mode, weight_mode, num_dimensions, num_points, seed):
    """Objective function maximizing output variance, while punishing weights.

    Parameters
    ----------
    individual : Individual
        Individual of the Cartesian Genetic Programming Framework.
    mode (string): Defines the first term of the fitness function
        options: 'variance', 'angle'
    weight_mode (scalar):  : Defines the way of calculating the weight penalty term
        options:
            '1': calculates the diff to a squared norm of 1
            '0': calculates the squared sum of the
    num_dimensions (scalar): input data dimensionality
    num_points (scalar): number of input data points
    rng: np.random.RandomState
    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """

    max_var_input = 1
    alpha = num_dimensions * max_var_input

    if individual.fitness is not None:
        return individual

    # Todo: This is presumably the wrong place create the data
    input_data = create_input_data(num_points, num_dimensions, max_var_input, seed)

    current_learning_rule = individual.to_numpy()

    [weights, generation_output] = learn_weights(input_data, learning_rule=current_learning_rule)

    weight_penalty = calc_weight_penalty(weights[-1, :], weight_mode)
    if mode == "variance":
        first_term = np.var(generation_output[int(0.9*num_points):,:]) # for validation use only the last 100 elements
    elif mode == "angle":
        first_term = calc_difference_to_first_pc(input_data, weights[-1,:])

    fitness = first_term - alpha * weight_penalty
    if (np.isnan(fitness) or weight_penalty > 1000):
        individual.fitness = -np.inf
    else:
        individual.fitness = fitness

    individual.weights = weights[-1,:]
    return individual


def evolution(population_params, genome_params, ea_params, evolve_params, seed):
    """Execute CGP for given target function.

    Parameters
    ----------
    target_function : Callable Target function
    population_params: dict with n_parents, mutation_rate, seed
    genome_params: dict with  n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives (allowed function gene values)
    ea_params: dict with n_offsprings, n_breeding, tournament_size, n_processes,
    evolve_params: dict with max_generations, min_fitness
    seed:

    Returns
    -------
    dict
        Dictionary containing the history of the evolution
    Individual
        Individual with the highest fitness in the last generation
    """

    pop = cgp.Population(**population_params, genome_params=genome_params)
    ea = cgp.ea.MuPlusLambda(**ea_params)

    history = {}
    history["fitness_parents"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())

    obj = functools.partial(
        objective,
        mode="variance",
        weight_mode=1,
        num_dimensions=2,
        num_points=1000,
        seed=seed,
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion


if __name__ == "__main__":

    seed = 1234
    np.random.seed(seed)

    population_params = {"n_parents": 10, "mutation_rate": 0.1, "seed": seed}

    genome_params = {
        "n_inputs": 3,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "levels_back": 5,
        "primitives": (cgp.Add, cgp.Sub) #, cgp.Mul),  # cgp.Div,
    }

    ea_params = {
        "n_offsprings": 10,
        "n_breeding": 10,
        "tournament_size": 2,
        "n_processes": 2,
    }

    evolve_params = {"max_generations": 1000, "min_fitness": 1000.0}

    [history, champion] = evolution(population_params, genome_params, ea_params, evolve_params, seed)

    champion_genome = champion.genome
    temp_graph = cgp.CartesianGraph(champion_genome)
    champion_active_gene = temp_graph.determine_active_regions()
    champion_pretty_str = temp_graph.pretty_str()
    champion_function = champion.to_func()
