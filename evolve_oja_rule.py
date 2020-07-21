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


def calculate_fitness(current_learning_rule, datasets, alpha, mode, weight_mode, train_fraction):

    first_term = 0
    weight_penalty = 0
    weights_final_per_dataset = []

    for dataset in datasets:

        data_train = dataset[0:int(train_fraction * num_points), :]
        data_validate = dataset[int(train_fraction * num_points):, :]

        [weights, _] = learn_weights(data_train, learning_rule=current_learning_rule) # todo: if weights are returned as nan set fitness to -inf
        weights_final = weights[-1, :]
        weights_final_per_dataset.append(weights_final)

        weight_penalty += calc_weight_penalty(weights_final, weight_mode)

        output = evaluate_output(data_validate, weights_final)

        if mode == "variance":
            first_term += np.var(
                output
        )  # for validation use only the last 100 elements
        elif mode == "angle":
            Warning(
                "Fitness function hyperparam alpha is currently adapted to using variance - use this mode carefully"
            )
            first_term += calc_difference_to_first_pc(
                data_train, weights[-1, :]
            )  # Todo: use data_train or dataset?

    fitness = first_term - alpha * weight_penalty
    return fitness, weights_final_per_dataset


def objective(individual, datasets, alpha, mode, weight_mode):
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

    if individual.fitness is not None:
        return individual

    current_learning_rule = individual.to_numpy()
    train_fraction = 0.9

    fitness, weights_final_per_dataset = calculate_fitness(current_learning_rule, datasets, alpha, mode, weight_mode, train_fraction)

    individual.fitness = fitness
    individual.weights = weights_final_per_dataset
    return individual


def evolution(
    datasets, population_params, genome_params, ea_params, evolve_params, alpha
):
    """Execute CGP for given target function.

    Parameters
    ----------
    datasets : List of dataset(s)
    population_params: dict with n_parents, mutation_rate, seed
    genome_params: dict with  n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives (allowed function gene values)
    ea_params: dict with n_offsprings, n_breeding, tournament_size, n_processes,
    evolve_params: dict with max_generations, min_fitness
    alpha: Hyperparameter weighting the second term of the fitness function

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
    history["champion_dna"] = []
    history["weights_champion"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["champion_dna"].append(pop.champion.genome.dna) # Todo: check
        #history["weights_champion"].append(pop.champion.weights)  # Todo: why does this not work?
        #Todo -> The issue is with the offspring created in
        # https://github.com/Happy-Algorithms-League/hal-cgp/blob/5421d9cdf0812ab3098d54c201ee115fa3129bce/cgp/ea/mu_plus_lambda.py#L103

    obj = functools.partial(
        objective, datasets=datasets, alpha=alpha, mode="variance", weight_mode=1,
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion


if __name__ == "__main__":

    seed = 1234
    np.random.seed(seed)

    # data parameters
    n_datasets = 3
    num_dimensions = 2
    num_points = 1000
    max_var_input = 1

    # hyperparameter for weighting fitness function terms
    alpha = num_dimensions * max_var_input

    population_params = {"n_parents": 10, "mutation_rate": 0.1, "seed": seed}

    genome_params = {
        "n_inputs": 3,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "levels_back": 5,
        "primitives": (cgp.Add, cgp.Sub, cgp.Mul),  # cgp.Div,
    }

    ea_params = {
        "n_offsprings": 10,
        "tournament_size": 2,
        "n_processes": 2,
    }

    evolve_params = {
        "max_generations": 2,  # todo change back (1000)
        "min_fitness": 1000.0,
    }  # Todo: What does min fitness mean?

    # initialize datasets
    datasets = []
    for idx in range(n_datasets):
        dataset = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )
        datasets.append(dataset)

    # evaluate hypothetical fitness of oja rule
    oja_fitness, _ = calculate_fitness(oja_rule, datasets, alpha, mode="variance", weight_mode=1, train_fraction=0.9)

    [history, champion] = evolution(
        datasets, population_params, genome_params, ea_params, evolve_params, alpha
    )
    # Todo: create Genome from champion-dna to evaluate final weight vector

    champion_graph = cgp.CartesianGraph(champion.genome)
    champion_active_gene = champion_graph.determine_active_regions()


