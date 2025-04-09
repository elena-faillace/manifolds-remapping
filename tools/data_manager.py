"""This module contains functions to load data from the data folder, or handle paths."""

import os
os.sys.path.append(os.pardir)
import glob
import numpy as np
import pandas as pd

import params
root_dir = params.root_dir


def find_path_to_data_folder(animal, fov=None, experiment=None, run=None):
    """Given the animal, fov, experiment and run find the path to the data folder. Made to account the type.
    If experiment and run are not provided, it will return the path of the animal and fov."""
    if (experiment is not None) and (run is not None) and (fov is not None):
        res = glob.glob(f'{root_dir+'data/'}/**/m*/m*/', recursive=True)
        for path in res:
            animal_path = path.split('/')[-2].split('_')[0]
            fov_path = path.split('/')[-2].split('_')[1]
            experiment_path = path.split('/')[-2].split('_')[2].split('-')[0]
            run_path = path.split('/')[-2].split('_')[2].split('-')[1].split('.')[0]
            if (animal==animal_path) and (fov==fov_path) and (experiment==experiment_path) and (run==run_path):
                return path
        print('Not path found for: ', animal, fov, experiment, run)
    else:
        res = glob.glob(f'{root_dir+'data/'}/**/{animal}/', recursive=True)
        return res[0]
    return None

def find_path_to_csv(animal, fov, experiment, run):
    """Given the animal, fov, experiment and run find the path to the data to then load it."""
    res = glob.glob(f'{root_dir}data/**/m*_spikes.csv', recursive=True)
    for path in res:
        animal_path = path.split('/')[-1].split('_')[0]
        fov_path = path.split('/')[-1].split('_')[1]
        experiment_path = path.split('/')[-1].split('_')[2].split('-')[0]
        run_path = path.split('/')[-1].split('_')[2].split('-')[1].split('.')[0]
        if (animal==animal_path) and (fov==fov_path) and (experiment==experiment_path) and (run==run_path):
            return path
    print('Not path to csv found for: ', animal, fov, experiment, run, type)
    return None

def load_csv_data(animal, fov, experiment, run):
    """Load the .csv file I made with all the data from Ann. It can be the spikes or the 'traces'."""
    path_to_csv = find_path_to_csv(animal, fov, experiment, run)
    try:
        df = pd.read_csv(path_to_csv)
    except FileNotFoundError: 
        print('File not found: ' + path_to_csv)
    return df

def get_all_experiments_runs(animal, fov):
    """
    Get all the experiments-sessions for a given animal and fov. 
    Return a list of tuples (experiment, run).
    """
    res = glob.glob(f'{root_dir}data/**/{animal}/{animal}_{fov}*', recursive=True)
    # remove if it is not a folder
    res = [res[i] for i in range(len(res)) if os.path.isdir(res[i])]
    all_sessions = [tuple(res[i].split(f'{animal}_{fov}_')[1].split('-')) for i in range(len(res))]
    return all_sessions

def get_fovs_given_animal(animal):
    """Get the fovs for a given animal."""
    res = glob.glob(f'{root_dir}data/*/{animal}/*', recursive=True)
    fovs = np.unique([res[i].split('/')[-1].split('_')[1] for i in range(len(res))])
    return fovs

# # TODO: might delete if Ca traces are not used
# def load_ca_data(animal, fov, experiment, run):
#     """Load the calcium traces data."""
#     # Find the path
#     path = glob.glob(f'{root_dir}data/**/{animal}/{animal}_{fov}_{experiment}-{run}/{animal}_{fov}_{experiment}-{run}_traces.csv', recursive=True)
#     # Load the data
#     try:
#         df = pd.read_csv(path[0])
#     except FileNotFoundError: 
#         print('File not found: ' + path)
#     return df

def load_global_index(animal, fov):
    """Load the global index dataframe."""
    path = glob.glob(f'{root_dir}data/**/{animal}/{animal}_{fov}_global_index_ref.csv', recursive=True)[0]
    try:
        df = pd.read_csv(path)
    except FileNotFoundError: 
        print('File not found: ' + path)
    return df

