import numpy as np
import matplotlib.pyplot as plt
import functools
import cgp
import pickle
import sympy
import json
import sys
import os

from learning_rules import oja_rule
from functions import *
from evolution import evolution,  calculate_fitness


if __name__ == "__main__":

    seed_offset = int(sys.argv[1])

    with open('params.pickle', 'rb') as f:
        params = pickle.load(f)

    params['seed'] += seed_offset
    seed = params['seed']
    params['population_params']['seed'] = seed
    rng = np.random.RandomState(params['seed'])

    # extract data parameters
    data_params = params['data_params']
    n_datasets = data_params['n_datasets']
    num_dimensions = data_params['num_dimensions']
    num_points = data_params['num_points']
    max_var_input = data_params['max_var_input']
    train_fraction = 0.9  # hardcoded

    # extract learning & fitness parameters
    learning_rate = params["learning rate"]
    alpha = num_dimensions * max_var_input
    fitness_mode = params['fitness_mode']

    # initialize datasets and initial weights
    datasets = []
    pc0_per_dataset = []
    pc0_empirical_per_dataset = []
    initial_weights_per_dataset = np.zeros([n_datasets, num_dimensions])

    for idx in range(n_datasets):
        dataset, cov_mat = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )

        data_train = dataset[0 : int(train_fraction * num_points), :]
        data_validate = dataset[int(train_fraction * num_points) :, :]

        datasets.append({
            "data_train": data_train,
            "data_validate": data_validate
        })

        initial_weights = initialize_weights(num_dimensions, rng)
        initial_weights_per_dataset[idx, :] = initial_weights

        pc0 = calculate_eigenvector_for_largest_eigenvalue(cov_mat)
        pc0_per_dataset.append(pc0)
        pc0_empirical = compute_first_pc(dataset)
        pc0_empirical_per_dataset.append(pc0_empirical)

    [history, champion] = evolution(
        datasets, pc0_per_dataset, initial_weights_per_dataset, params['population_params'],
        params['genome_params'], params['ea_params'], params['evolve_params'],
        learning_rate, alpha, fitness_mode)

    # evaluate weights of champion (not passed down for non-re-evaluated champion)
    champion_learning_rule = cgp.CartesianGraph(champion.genome).to_numpy()
    champion_fitness, champion_weights_per_dataset = calculate_fitness(
        champion_learning_rule,
        datasets,
        pc0_per_dataset,
        initial_weights_per_dataset,
        learning_rate,
        alpha,
        fitness_mode,
    )

    # evaluate hypothetical fitness of oja rule
    oja_fitness, oja_weights_per_dataset = calculate_fitness(
        oja_rule, datasets, pc0_per_dataset, initial_weights_per_dataset, learning_rate, alpha, fitness_mode)

    save_data = {'param' : params,
                     'champion':  {
                         'champion': champion,
                         'champion_fitness' : champion_fitness,
                         'champion_weights' : champion_weights_per_dataset
                         # Todo: will be removable once cgp #222 is merged
                     },
                    'oja': {
                        'oja_fitness': oja_fitness,
                        'oja_weights': oja_weights_per_dataset
                    },
                     'history' : history,
                     }

    filename = os.path.join(params['outputdir'], 'data' + str(seed_offset) +'.pickle')
    data_file = open(filename, 'wb')
    pickle.dump(save_data, data_file)
