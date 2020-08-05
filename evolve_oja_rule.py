import numpy as np
import matplotlib.pyplot as plt
import functools
import cgp
import pickle
import sympy

from learning_rules import *
from functions import *


def f_target(x, w, y):
    oja = y * (x - y * w)
    return oja


def calculate_fitness(
    current_learning_rule, datasets, pc0_per_dataset, alpha, mode, weight_mode, train_fraction, rng
):

    first_term = 0
    weight_penalty = 0
    weights_final_per_dataset = []

    for idx_dataset, dataset in enumerate(datasets):

        data_train = dataset[0 : int(train_fraction * num_points), :]  #This is repeated in every generation which is unnecessary
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

    obj = functools.partial(
        objective, datasets=datasets, pc0_per_dataset=pc0_per_dataset, alpha=alpha, mode=fitness_mode,
        weight_mode=1, rng=rng
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion


if __name__ == "__main__":

    seed = 10000
    rng = np.random.RandomState(seed)
    flag_save_figures = True
    flag_save_data = True

    # data parameters
    n_datasets = 3
    num_dimensions = 2
    num_points = 5000
    max_var_input = 1

    # fitness parameters
    fitness_mode = "variance"
    alpha = num_dimensions * max_var_input  # hyperparameter for weighting fitness function terms

    population_params = {"n_parents": 10, "mutation_rate": 0.1, "seed": seed}

    genome_params = {
        "n_inputs": 3,
        "n_outputs": 1,
        "n_columns": 10,
        "n_rows": 2,
        "levels_back": 5,
        "primitives": (cgp.Sub, cgp.Mul),  # cgp.Add, cgp.Div
    }

    ea_params = {
        "n_offsprings": 10,
        "tournament_size": 2,
        "n_processes": 2,
    }

    evolve_params = {
        "max_generations": 1000,
        "min_fitness": 1000.0,
    }  #

    # initialize datasets
    datasets = []
    pc0_per_dataset = []

    for idx in range(n_datasets):
        dataset = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )
        datasets.append(dataset)
        pc0 = compute_first_pc(dataset)
        pc0_per_dataset.append(pc0)

    # Todo: check if there has to be some reset within the rng
    [history, champion] = evolution(
        datasets, pc0_per_dataset, population_params, genome_params, ea_params, evolve_params, alpha, fitness_mode, rng
    )
    rng.seed(seed)
    champion_learning_rule = cgp.CartesianGraph(champion.genome).to_numpy()
    champion_fitness, champion_weights_per_dataset = calculate_fitness(
        champion_learning_rule,
        datasets,
        pc0_per_dataset,
        alpha,
        mode=fitness_mode,
        weight_mode=1,
        train_fraction=0.9,
        rng=rng
    )
    champion_sympy_expression = champion.to_sympy()

    # evaluate hypothetical fitness of oja rule
    rng.seed(seed)
    oja_fitness, oja_weights_per_dataset = calculate_fitness(
        oja_rule, datasets, pc0_per_dataset, alpha, mode="variance", weight_mode=1, train_fraction=0.9,
    rng=rng)

    # plot (works only for n_dimensions = 2 at the moment)
    m = np.linspace(-1, 1, 1000)
    champion_angle = np.zeros(n_datasets)
    oja_angle = np.zeros(n_datasets)

    for idx in range(n_datasets):

        temp_champ_angle = compute_angle_weight_first_pc(
            champion_weights_per_dataset[idx], pc0_per_dataset[idx], mode="degree"
        )
        champion_angle[idx] = calculate_smallest_angle(temp_champ_angle)
        temp_oja_angle = compute_angle_weight_first_pc(
            oja_weights_per_dataset[idx], pc0_per_dataset[idx], mode="degree"
        )
        oja_angle[idx] = calculate_smallest_angle(temp_oja_angle)

        champion_as_line = np.zeros([num_dimensions, np.size(m)])
        champion_as_line[0, :] = champion_weights_per_dataset[idx][0] * m
        champion_as_line[1, :] = champion_weights_per_dataset[idx][1] * m

        oja_as_line = np.zeros([num_dimensions, np.size(m)])
        oja_as_line[0, :] = oja_weights_per_dataset[idx][0] * m
        oja_as_line[1, :] = oja_weights_per_dataset[idx][1] * m

        pc0_as_line = np.zeros([num_dimensions, np.size(m)])
        pc0_as_line[0, :] = pc0_per_dataset[idx][0] * m
        pc0_as_line[1, :] = pc0_per_dataset[idx][1] * m

        # Todo: set learning rule as eg. plot title
        fig, ax = plt.subplots()
        plt.grid()
        plt.plot(champion_as_line[0, :], champion_as_line[1, :])
        plt.plot(oja_as_line[0, :], oja_as_line[1, :])
        plt.plot(pc0_as_line[0, :], pc0_as_line[1, :])
        plt.xlim(-3, 3)
        plt.ylim(-3, 3)
        plt.xlabel("w_0")
        plt.ylabel("w_1")
        plt.legend(
            [
                "champion angle:" + str(champion_angle[idx]),
                "oja angle: " + str(oja_angle[idx]),
                "true pc 1",
            ]
        )

        if flag_save_figures:
            fig.savefig("figures/weight_vectors_seed" + str(seed+idx) + ".png")

        if flag_save_data:
            param_list = [ea_params, evolve_params, genome_params, population_params, seed, n_datasets, num_dimensions,
                            num_points, max_var_input, fitness_mode]

            save_data_list = [param_list, champion, history, champion_sympy_expression]
            # sympy expression purely for convenience
            data_file = open('data/data_seed' + str(seed+idx) + '.pickle', 'wb')
            pickle.dump(save_data_list, data_file)


