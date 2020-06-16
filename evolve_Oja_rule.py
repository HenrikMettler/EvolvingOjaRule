import torch
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
        out = abs(np.sum(np.square(weights)) - 1)
    elif mode == 0:
        out = np.sqrt(np.sum(np.square(weights)))
    else:
        raise NotImplementedError(
            "Mode for calculating the weight penalty not available, choose 0 or 1"
        )
    return out


def objective(
    individual, mode, weight_mode, num_dimensions=2, num_points=1000, alpha=0.5, seed=1
):
    """Objective function maximizing output variance, while punishing wei.

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
    alpha (scalar): hyperparameter, weighting between weight norm deviation / var(y)
    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """

    if individual.fitness is not None:
        return individual

    current_learning_rule = individual.to_numpy()
    input_data = create_input_data(num_points, num_dimensions)
    [weights, generation_output] = learn_weights(
        input_data, learning_rule=current_learning_rule
    )

    weight_penalty = calc_weight_penalty(weights[-1, :], weight_mode)
    if mode == "variance":
        first_term = np.var(generation_output)
    elif mode == "angle":
        # calc PC1 of dataset
        data_first_pc = compute_first_pc(input_data)
        # calc angle PC1 to w
        angles = compute_angle_weights_first_pc(weights, data_first_pc)
        angle = angles[-1]
        # calc diff to Pi
        first_term = np.minimum(angle, np.pi - angle)
        # * - 1

    fitness = -(1 - alpha) * first_term - alpha * weight_penalty
    if (
        np.isnan(fitness) or weight_penalty > 1000
    ):  # don't allow too large weights Todo: (keep in mind)
        individual.fitness = -np.inf
    else:
        individual.fitness = fitness

    return individual


def evolution(
    target_function, population_params, genome_params, ea_params, evolve_params
):
    """Execute CGP for given target function.

    Parameters
    ----------
    target_function : Callable Target function
    population_params: dict with n_parents, mutation_rate, seed
    genome_params: dict with  n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives (allowed function gene values)
    ea_params: dict with n_offsprings, n_breeding, tournament_size, n_processes,
    evolve_params: dict with max_generations, min_fitness

    Returns
    -------
    dict
        Dictionary containing the history of the evolution
    Individual
        Individual with the highest fitness in the last generation
    """

    # create population that will be evolved
    pop = cgp.Population(**population_params, genome_params=genome_params)

    # create instance of evolutionary algorithm
    ea = cgp.ea.MuPlusLambda(**ea_params)

    # define callback for recording of fitness over generations
    history = {}
    history["fitness_parents"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())

    # the objective passed to evolve should only accept one argument,
    # the individual
    obj = functools.partial(
        objective,
        mode="angle",
        weight_mode=1,
        num_dimensions=2,
        num_points=1000,
        seed=population_params["seed"],
    )

    # Perform the evolution
    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion


if __name__ == "__main__":

    population_params = {"n_parents": 10, "mutation_rate": 0.5, "seed": 2}

    genome_params = {
        "n_inputs": 3,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "levels_back": 5,
        "primitives": (cgp.Add, cgp.Sub, cgp.Mul, cgp.ConstantFloat),  # cgp.Div,
    }

    ea_params = {
        "n_offsprings": 10,
        "n_breeding": 10,
        "tournament_size": 2,
        "n_processes": 2,
    }

    evolve_params = {"max_generations": 1000, "min_fitness": 1000.0}

    [history, champion] = evolution(
        f_target, population_params, genome_params, ea_params, evolve_params
    )

    a = 1
    champion_genome = champion.genome
    temp_graph = cgp.CartesianGraph(champion_genome)
    champion_active_gene = temp_graph.determine_active_regions()
    champion_pretty_str = temp_graph.pretty_str()
    champion_function = champion.to_func()
