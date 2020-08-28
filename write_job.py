import dicthash
import json
import numpy as np
import os
import sys

import cgp

sys.path.insert(0, '../../includes/')
import write_job_utils as utils


if __name__ == '__main__':

    params = {

        # machine setup
        'submit_command': 'sbatch',
        'jobfile_template': 'juwels_template.jdf',
        'jobname': 'evolving-oja-rule',
        'wall_clock_limit': '24:00:00',
        'ntasks': 6,
        'cpus-per-task': 8,
        'n_nodes': 1,
        'mail-user': 'henrik.mettler@gmail.com',
        'account': 'hhd34',
        'partition': 'batch',
        'sim_script': 'juwels_main.py',
        'dependencies': ['learning_rules.py', 'functions.py', 'evolution.py', 'Ind_oja_min_length.pickle'],

        'seed':  1,

        'data_params' : {
            'n_datasets' : 10,
            'num_dimensions' : 2,
            'num_points' : 2000,
            'max_var_input' : 1,
        },

        'fitness_mode' : "variance",
        'learning rate' : 0.01,  # Todo: check lr with oja

        'population_params': {
            'n_parents': 1,
            'mutation_rate': 0.05,
        },

        'genome_params' : {
            'n_inputs': 3,
            'n_outputs': 1,
            'n_columns': 1000,  # Todo check values
            'n_rows': 1,
            'levels_back': None,
            'primitives': (cgp.Sub, cgp.Mul),
        },

        'ea_params' : {
            'n_offsprings': 4,
            'tournament_size': 2,
            'n_processes': 8,
        },

        'evolve_params' : {
        'max_generations': 2000,
        'min_fitness': 9.99,
        },

    }

    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script'])  # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']]  # consistency check

    mutation_rates = [0.01, 0.02, 0.05, 0.1, 0.2]

    for mutation_rate in mutation_rates:

        params['population_params']['mutation_rate'] = mutation_rate

        key = dicthash.generate_hash_from_dict(params)

        params['outputdir'] = os.path.join(os.getcwd(), key)
        params['workingdir'] = os.getcwd()

        submit_job = True

        print('preparing job')
        print(' ', params['outputdir'])

        utils.mkdirp(params['outputdir'])
        utils.write_pickle(params, os.path.join(params['outputdir'], 'params.pickle'))
        utils.create_jobfile(params)
        utils.copy_file(params['sim_script'], params['outputdir'])
        utils.copy_files(params['dependencies'], params['outputdir'])
        if submit_job:
            print('submitting job')
            utils.submit_job(params)
