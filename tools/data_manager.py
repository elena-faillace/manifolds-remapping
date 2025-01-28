"""This module contains functions to load data from the data folder, or handle paths."""

import os
os.sys.path.append(os.pardir)
import glob
import numpy as np
import pandas as pd

import global_vars
root_dir = global_vars.root_dir


def find_path_to_data_folder(animal, fov, experiment, run):
    """Given the animal, fov, experiment and run find the path to the data folder."""
    res = glob.glob(f'{root_dir}data/**/m*/m*/', recursive=True)
    for path in res:
        animal_path = path.split('/')[-2].split('_')[0]
        fov_path = path.split('/')[-2].split('_')[1]
        experiment_path = path.split('/')[-2].split('_')[2].split('-')[0]
        run_path = path.split('/')[-2].split('_')[2].split('-')[1].split('.')[0]
        if (animal==animal_path) and (fov==fov_path) and (experiment==experiment_path) and (run==run_path):
            return path
    print('Not path found for: ', animal, fov, experiment, run)
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
    all_sessions = [tuple(res[i].split(f'{animal}_{fov}_')[1].split('-')) for i in range(len(res))]
    return all_sessions

