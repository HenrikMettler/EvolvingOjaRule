import dicthash
import json
import numpy as np
import os
import sys

sys.path.insert(0, os.path.expanduser('~/work/projects/python-gp'))  # Todo: here
import cgp

sys.path.insert(0, '../../includes/')
import juwels.write_job_utils as utils


if __name__ == '__main__':

    #sim_key = dicthash.generate_hash_from_dict(sim_params)

    params = {

        #'sim_key': sim_key, # consistency check  # Todo check if sim_key can be removed

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
        'dependencies': ['../learning_rules.py', '../functions.py'],

        'seed':  1,

        'flag_save_figures': True,
        'flag_save_data' : True,
        'data_params' : {
            'n_datasets' : 10,
            'num_dimensions' : 2,
            'num_points' : 5000,
            'max_var_input' : 1,
        },


        'fitness_mode' : "angle",


        'population_params': {
            'n_parents': 1,
            'mutation_rate': 0.05,
        },

        'genome_params' : {
            'n_inputs': 3,
            'n_outputs': 1,
            'n_columns': 100,
            'n_rows': 1,
            'levels_back': None,
            'primitives': (cgp.Sub, cgp.Mul),  # cgp.Add, cgp.Div
        },

        'ea_params' : {
            'n_offsprings': 4,
            'tournament_size': 2,
            'n_processes': 8,
        },

        'evolve_params' : {
        'max_generations': 10000,
        'min_fitness': 1000.0,
        },

    }
    params['md5_hash_sim_script'] = utils.md5_file(params['sim_script']) # consistency check
    params['md5_hash_dependencies'] = [utils.md5_file(fn) for fn in params['dependencies']] # consistency check

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
