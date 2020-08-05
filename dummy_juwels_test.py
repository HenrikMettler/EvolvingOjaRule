import numpy as np
import matplotlib.pyplot as plt
import functools
#import cgp
import pickle
import sympy
import sys

seed = 1000
seed_offset = int(sys.argv[1])
seed = seed + seed_offset

# print the imported system variable
print(seed_offset, seed)

"""# create a cgp population and evolutionary algorithm (step 0)
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
}

pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)
"""


