import numpy as np
import matplotlib.pyplot as plt
import functools
import cgp
import pickle
import sympy
import json
import sys

from learning_rules import *
from functions import *

if __name__ == "__main__":

    seed_offset = int(sys.argv[1])

    with open('params.json', 'r') as f:
        params = json.load(f)
    params['seed'] += seed_offset
    seed = params['seed']
    print(seed)
    """rng = np.random.RandomState(params['seed'])

    params['population_params']['seed'] = seed

    # extract data parameters
    data_params = params['data_params']
    n_datasets = data_params['n_datasets']
    num_dimensions = data_params['num_dimensions']
    num_points = data_params['num_points']
    max_var_input = data_params['max_var_input']

    # initialize datasets
    datasets = []
    pc0_per_dataset = []

    for idx in range(data_params['n_datasets']):
        dataset, cov_mat = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )
        datasets.append(dataset)

        pc0 = compute_first_pc(dataset)
        pc0_per_dataset.append(pc0)

    # initialize fitness parameters
    alpha = num_dimensions * data_params['max_var_input']
    fitness_mode = params['fitness_mode']

    # Todo: check if there has to be some reset of the rng
    [history, champion] = evolution(
        datasets, pc0_per_dataset, params['population_params'],
        params['genome_params'], params['ea_params'], params['evolve_params'], alpha, fitness_mode, rng
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

        if params['flag_save_figures']:
            fig.savefig("figures/weight_vectors_seed" + str(seed+idx) + ".png")

    if params['flag_save_data']:

        save_data = {'param' : params,
                     'champion':  champion,
                     'history' : history,
                     'champion_sympy' : champion_sympy_expression
                     }

        # sympy expression purely for convenience
        data_file = open('data/data_seed' + str(seed+idx) + '.pickle', 'wb')
        pickle.dump(save_data, data_file)"""
