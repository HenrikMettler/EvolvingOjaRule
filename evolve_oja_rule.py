import numpy as np
import matplotlib.pyplot as plt
import functools
import cgp
import pickle
import sympy

from evolution import *
from learning_rules import oja_rule
from functions import *


def f_target(x, w, y):
    oja = y * (x - y * w)
    return oja


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

    # learning parameters
    learning_rate = 0.005

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
        "max_generations": 2,  # Todo: set to larger value
        "min_fitness": 1000.0,
    }  #

    # initialize datasets
    datasets = []
    pc0_per_dataset = []
    pc0_empirical_per_dataset = []
    initial_weights_per_dataset = []

    for idx in range(n_datasets):
        dataset, cov_mat = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )
        datasets.append(dataset)

        initial_weights = initialize_weights(num_dimensions, rng)
        initial_weights_per_dataset.append(initial_weights)

        pc0 = calculate_eigenvector_for_largest_eigenvalue(cov_mat)
        pc0_per_dataset.append(pc0)
        pc0_empirical = compute_first_pc(dataset)
        pc0_empirical_per_dataset.append(pc0_empirical)

    [history, champion] = evolution(
        datasets, pc0_per_dataset, initial_weights_per_dataset,
        population_params, genome_params, ea_params, evolve_params,
        learning_rate, alpha, fitness_mode)
    rng.seed(seed)
    champion_learning_rule = cgp.CartesianGraph(champion.genome).to_numpy()
    champion_fitness, champion_weights_per_dataset = calculate_fitness(
        champion_learning_rule,
        datasets,
        pc0_per_dataset,
        initial_weights_per_dataset,
        learning_rate, alpha, fitness_mode)
    champion_sympy_expression = champion.to_sympy()

    # evaluate hypothetical fitness of oja rule
    rng.seed(seed)
    oja_fitness, oja_weights_per_dataset = calculate_fitness(
        oja_rule, datasets, pc0_per_dataset, initial_weights_per_dataset, learning_rate, alpha, fitness_mode)

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

        save_data = {'params': {
                        'ea_params': ea_params,
                        'evolve_params' : evolve_params,
                        'genome_params' : genome_params,
                        'population_params' : population_params,
                        'seed' : seed,
                        'n_datasets' : n_datasets,
                        'num_dimensions' : num_dimensions,
                        'num_points' : num_points,
                        'max_var_input' : max_var_input,
                        'fitness_mode' : fitness_mode,
                            },
                     'champion': champion,
                     'history': history,
                     'champion_sympy': champion_sympy_expression
                     }
        # sympy expression purely for convenience
        data_file = open('data/data_seed' + str(seed) + '.pickle', 'wb')
        pickle.dump(save_data, data_file)


