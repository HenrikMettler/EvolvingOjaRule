import numpy as np
import cgp
import functools

from functions import *


#@cgp.utils.disk_cache(
#    "fec_caching.pkl",
#    use_fec=True,
#    fec_seed=12345,
 #   fec_min_value=-10.0,
 #   fec_max_value=10.0,
   # fec_batch_size=5,
#)
def calculate_fitness(individual, data, learning_rate,
                      alpha, mode):

    # hardcoded parameter
    weight_mode = 1  # '1': diff to a squared norm of 1 ;'0': squared sum of the weights

    # extract learning_rule
    current_learning_rule = individual.to_numpy()

    # initialize parameters
    first_term, weight_penalty = 0, 0
    weights_final_per_dataset = []

    for current_data in data:

        [weights, _] = learn_weights(current_data['data_train'], learning_rule=current_learning_rule,
                                     initial_weights=current_data['initial_weights'],
                                     learning_rate=learning_rate)
        weights_final = weights[-1, :]
        if np.any(np.isnan(weights_final)):
            weights_final = -np.inf * np.ones(np.shape(weights_final))
            weights_final_per_dataset.append(weights_final)
            return -np.inf, weights_final_per_dataset

        weight_penalty += calc_weight_penalty(weights_final, weight_mode)

        if mode == "variance":
            output = evaluate_output(current_data['data_validate'], weights_final)
            first_term += np.var(output)

        elif mode == "angle":
            if np.linalg.norm(weights_final) == 0:  # can not calculate angle for 0 - norm weights
                first_term = 0
            else:
                angles = compute_angles_weights_first_pc(weights, current_data['pc0'])
                angles_abs_cos = abs(np.cos(angles))
                first_term += np.mean(angles_abs_cos)

        weights_final_per_dataset.append(weights_final)

    fitness = first_term - alpha * weight_penalty

    return fitness, weights_final_per_dataset


def objective(individual, data, learning_rate, alpha, mode):
    """Objective function maximizing fitness (by maximizing variance or minimizing angle to PC0)
      while punishing large weights.

    Parameters
    ----------
    individual : Individual
        Individual of the Cartesian Genetic Programming Framework.
    data: List[dict]
    learning_rate: float
    alpha: relative weighting of the weight penalty term
    mode (string): Defines the first term of the fitness function options: 'variance', 'angle'

    Returns
    -------
    Individual
        Modified individual with updated fitness value.
    """

    if individual.fitness is not None:
        return individual

    fitness, weights_final_per_dataset = calculate_fitness(
        individual, data, learning_rate, alpha, mode)

    individual.fitness = fitness
    individual.weights = weights_final_per_dataset
    return individual


def evolution(data, population_params, genome_params, ea_params, evolve_params,
              learning_rate, alpha, fitness_mode):
    """Execute CGP for given target function.

    Parameters
    ----------
    data: List(dict)
     Fields in dict:
        data_train: np.array
        data_validate: np.array
        initial_weights: np.array
            -> pre defined so that every individual (resp learning rule) has the same starting condition
        pc0: First principal components (Eigenvectors of the co-variance matrix)
    population_params: dict with n_parents, mutation_rate, seed
    genome_params:
        dict with  n_inputs, n_outputs, n_columns, n_rows, levels_back, primitives (allowed function gene values)
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
    history["champion_sympy_expression"] = []

    def recording_callback(pop):
        history["fitness_parents"].append(pop.fitness_parents())
        history["champion_sympy_expression"].append(
            pop.champion.to_sympy()
        )

    obj = functools.partial(
        objective, data=data, learning_rate=learning_rate,
        alpha=alpha, mode=fitness_mode)

    cgp.evolve(
        pop, obj, ea, **evolve_params, print_progress=True, callback=recording_callback,
    )
    return history, pop.champion