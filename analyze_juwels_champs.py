import pickle
import numpy as np
import sympy
import matplotlib.pyplot as plt
import os

from sympy import symbols
from sympy.utilities.lambdify import lambdify

from functions import *
from learning_rules import oja_rule

fig_idx = 1
num_points = 2000

data_directory = 'data/x_init_5'  # Todo: adapt data creation to the way data is created if x_init_5?
data_filename = 'd2c1fc2fa6ca720aa037a795e445ffef'
data_dict = {}
for filename in os.listdir(f"{data_directory}/{data_filename}"):
    if filename.endswith('.pickle') & filename.startswith('data'):
        data_name = os.path.splitext(filename)[0]
        with open(f"{data_directory}/{data_filename}/{filename}", 'rb') as f:
            data_dict[data_name] = pickle.load(f)


# Create 4 datasets: 2 large first weight, 2 mixed and initial weight vectors
interesting_seeds = [4, 8, 6, 7]  # 4,8 large 1st, 6,7 mixed
datasets = []
pc0_per_dataset = []
initial_weights_per_dataset = [np.array([1/np.sqrt(2), 1/np.sqrt(2)]), np.array([1/np.sqrt(2), 1/np.sqrt(2)]),
                               np.array([0.0, 1.0]), np.array([0.0, 1.0])]  # set to be far away from target values
for seed in interesting_seeds:
    dataset, cov_mat = create_input_data(
        num_points=num_points, num_dimensions=2, max_var_input=1, seed=seed)  # Todo: this info is all also in the dict
    datasets.append(dataset)
    pc0 = calculate_eigenvector_for_largest_eigenvalue(cov_mat)
    pc0_per_dataset.append(pc0)


interesting_fitness_threshold = 7
for data_name in data_dict:
    fitness_parents = np.array(data_dict[data_name]['history']['fitness_parents'])
    champion_sympy = data_dict[data_name]['history']['champion_sympy_expression']
    oja_fitness = data_dict[data_name]['oja']['oja_fitness']
    fitness_parents_interesting = []
    champion_sympy_interesting = []
    for idx, fitness in enumerate(fitness_parents):
        if fitness > interesting_fitness_threshold:
            if len(fitness_parents_interesting) == 0 or fitness != fitness_parents_interesting[-1]:
                fitness_parents_interesting.append(fitness)
                champion_sympy_interesting.append(champion_sympy[idx])

    # Train for all expressions with interesting fitness values
    if len(fitness_parents_interesting) != 0:
        for champ_expression, fitness in zip(champion_sympy_interesting, fitness_parents_interesting):
            learning_rule_expression = replace_expression(champ_expression[0])
            learning_rule_as_function = create_function_from_expression(learning_rule_expression)

            fig = plt.figure(fig_idx, figsize=[15, 10])

            for idx_dataset, dataset in enumerate(datasets):

                [champion_weights, _] = learn_weights_with_lambdify(dataset, learning_rule_as_lambdify=learning_rule_as_function,
                                             initial_weights=initial_weights_per_dataset[idx_dataset],
                                             learning_rate=data_dict[data_name]['param']['learning rate'])
                [oja_weights, _] = learn_weights(dataset, learning_rule=oja_rule,
                                             initial_weights=initial_weights_per_dataset[idx_dataset],
                                             learning_rate=data_dict[data_name]['param']['learning rate'])

                champion_angle = np.zeros(num_points)
                oja_angle = np.zeros(num_points)
                for idx_point in range(num_points):
                    champion_angle[idx_point] = compute_angle_weight_first_pc(champion_weights[idx_point],
                                                                              pc0_per_dataset[idx_dataset])
                    champion_angle[idx_point] = np.min([np.pi-champion_angle[idx_point], champion_angle[idx_point]])
                    oja_angle[idx_point] = compute_angle_weight_first_pc(oja_weights[idx_point],
                                                                         pc0_per_dataset[idx_dataset])
                    oja_angle[idx_point] = np.min([np.pi-oja_angle[idx_point], oja_angle[idx_point]])

                plt.subplot(2,2,idx_dataset+1)
                plt.plot(champion_angle)
                plt.plot(oja_angle)
                plt.xlim(-50, 2000 + 50)
                plt.ylim(-0.2, np.pi/2)
                plt.xlabel("sample")
                plt.ylabel("angle ")
                plt.legend([f"{learning_rule_expression}:  {round(fitness[0],4)}", f"oja {round(oja_fitness,4)}"])
                plt.title("Init_w: " + str(initial_weights_per_dataset[idx_dataset])
                                           + "  first PC: " + str(pc0_per_dataset[idx_dataset]))

            fig_name = f"{data_directory}/{data_filename}/{learning_rule_expression}_{data_name}.png"
            plt.savefig(fig_name)
            plt.close()
            fig_idx+=1
