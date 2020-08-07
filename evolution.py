import numpy as np
import cgp
import functools

from functions import *


def calculate_fitness(
    current_learning_rule, datasets, pc0_per_dataset, alpha, mode, weight_mode, train_fraction, rng
):

    first_term = 0
    weight_penalty = 0
    weights_final_per_dataset = []
    num_points = np.size(datasets[0],0)

    for idx_dataset, dataset in enumerate(datasets):

        data_train = dataset[0 : int(train_fraction * num_points), :]
        data_validate = dataset[int(train_fraction * num_points) :, :]

        [weights, _] = learn_weights(data_train, learning_rule=current_learning_rule, rng=rng)
        weights_final = weights[-1, :]
        if np.any(np.isnan(weights_final)):
            weights_final = -np.inf * np.ones(np.shape(weights_final))
            weights_final_per_dataset.append(weights_final)
            return -np.inf, weights_final_per_dataset

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
            angle = compute_angle_weight_first_pc(weights_final, pc0_per_dataset[idx_dataset])
            first_term += abs(np.cos(angle))

    fitness = first_term - alpha * weight_penalty
    return fitness, weights_final_per_dataset


def objective(individual, datasets, pc0_per_dataset, alpha, mode, weight_mode, rng):
    """Objective function maximizing fitness (by maximizing variance or minimizing angle to PC0)
      while punishing weights.

    Parameters
    ----------
    individual : Individual
        Individual of the Cartesian Genetic Programming Framework.
    datasets: List of dataset for the individual
    pc0_per_dataset: List of first PC for every dataset
    alpha: relative weighting of the weight penalty term
    mode (string): Defines the first term of the fitness function options: 'variance', 'angle'
    weight_mode (scalar):  : Defines the way of calculating the weight penalty term options:
            '1': calculates the diff to a squared norm of 1
            '0': calculates the squared sum of the weights
   rng : numpy.RandomState
            Random number generator instance to use for randomizing.

    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """

    if individual.fitness is not None:
        return individual

    current_learning_rule = individual.to_numpy()
    train_fraction = 0.9

    fitness, weights_final_per_dataset = calculate_fitness(
        current_learning_rule, datasets, pc0_per_dataset, alpha, mode, weight_mode, train_fraction, rng
    )

    individual.fitness = fitness
    individual.weights = weights_final_per_dataset
    return individual


def evolution(
    datasets, pc0_per_dataset, population_params, genome_params, ea_params, evolve_params, alpha, fitness_mode, rng
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
    fitness_mode: str ("angle" or "variance")

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
    history["champion_genome"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["champion_genome"].append(
            pop.champion.genome
        )  # use genome not dna to enable use of CartesianGraph(genome).to_numpy()
        # history["weights_champion"].append(pop.champion.weights)

    # Todo: check if there has to be some reset of the rng
    obj = functools.partial(
        objective, datasets=datasets, pc0_per_dataset=pc0_per_dataset, alpha=alpha, mode=fitness_mode,
        weight_mode=1, rng=rng
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion