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


def calculate_fitness(
    current_learning_rule, datasets, alpha, mode, weight_mode, train_fraction
):

    first_term = 0
    weight_penalty = 0
    weights_final_per_dataset = []

    for dataset in datasets:

        data_train = dataset[0 : int(train_fraction * num_points), :]
        data_validate = dataset[int(train_fraction * num_points) :, :]

        [weights, _] = learn_weights(data_train, learning_rule=current_learning_rule)
        weights_final = weights[-1, :]
        if np.any(np.isnan(weights_final)):
            weights_final = -np.inf * np.ones(np.shape(weights_final))
            # Todo: we could return the loop here as well since this will set the fitness overall to -np.inf
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
            '0': calculates the squared sum of the weights
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

    fitness, weights_final_per_dataset = calculate_fitness(
        current_learning_rule, datasets, alpha, mode, weight_mode, train_fraction
    )

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
    history["champion_genome"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["champion_genome"].append(
            pop.champion.genome
        )  # use genome not dna to enable use of CartesianGraph(genome).to_numpy()
        # history["weights_champion"].append(pop.champion.weights)  Todo: Does not work since offspring (cloned) are not reevaluated (and additional properties are not passed down)

    obj = functools.partial(
        objective, datasets=datasets, alpha=alpha, mode="variance", weight_mode=1,
    )

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion


if __name__ == "__main__":

    seed = 1000  # before 1234
    np.random.seed(seed)
    flag_save_figures = True

    # data parameters
    n_datasets = 5
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
        "max_generations": 500,
        "min_fitness": 1000.0,
    }  #

    # initialize datasets
    datasets = []
    pc0_per_dataset = []
    for idx in range(n_datasets):
        dataset = create_input_data(
            num_points, num_dimensions, max_var_input, seed + idx
        )
        pc0 = compute_first_pc(dataset)
        pc0_per_dataset.append(pc0)
        datasets.append(dataset)

    [history, champion] = evolution(
        datasets, population_params, genome_params, ea_params, evolve_params, alpha
    )
    # Todo: do for all champions in history?
    champion_learning_rule = cgp.CartesianGraph(champion.genome).to_numpy()
    champion_fitness, champion_weights_per_dataset = calculate_fitness(
        champion_learning_rule,
        datasets,
        alpha,
        mode="variance",
        weight_mode=1,
        train_fraction=0.9,
    )

    # evaluate hypothetical fitness of oja rule
    oja_fitness, oja_weights_per_dataset = calculate_fitness(
        oja_rule, datasets, alpha, mode="variance", weight_mode=1, train_fraction=0.9
    )

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
            # Todo: implement automatic saving
            a = 1
