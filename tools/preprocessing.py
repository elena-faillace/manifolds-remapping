"""This script contains functions to preprocess the data before the analysis."""

import os
from sys import exception
from scipy.io import loadmat
import numpy as np
import pandas as pd
from itertools import groupby

import params

path_to_rois = params.path_to_rois
sampling_freq = params.sampling_freq

from tools.data_manager import find_path_to_data_folder, find_path_to_csv, load_csv_data


# Combine the files from .mat into .csv


def combine_files_to_csv(animal, fov, experiment, run):
    """Combine the raw files into a .csv file with the information I might need later."""
    path_to_data = find_path_to_data_folder(animal, fov, experiment, run)
    print(
        "...start loading the files for: "
        + animal
        + "_"
        + fov
        + "_"
        + experiment
        + "-"
        + run
    )

    # Load the data
    beh_file, spikes_file = load_raw_files_to_combine(animal, fov, experiment, run)

    # Extract the data of behavior
    phi = beh_file["phi"].flatten()
    x = beh_file["x"].flatten()
    y = beh_file["y"].flatten()
    speed = beh_file["speed"].flatten()
    radius = beh_file["r"].flatten()
    time = beh_file["time"].flatten()
    # Extract the data of spikes
    events = spikes_file["spikes"]
    cell_ids = np.arange(events.shape[0]) + 1

    # Make a pandas dataframe with the data of the spikes
    events = pd.DataFrame(events.T, columns=[str(cell_id) for cell_id in cell_ids])

    # Add behavior data
    beh_df = pd.DataFrame(
        {
            "animal": animal,
            "experiment": experiment,
            "run": run,
            "fov": fov,
            "time": time,
            "phi": phi,
            "x": x,
            "y": y,
            "speed": speed,
            "radius": radius,
        }
    )

    # Combine the dataframes
    df_events = pd.concat([beh_df, events], axis=1)

    # The following adjustments are made based on individual recordings that I found out having a problem
    if (
        (animal == "m66")
        and (fov == "fov1")
        and (experiment == "fam1fam2")
        and (run == "fam1")
    ):
        max_lenght = len(time)
        df_events = df_events.iloc[:max_lenght, :]

    # Save the dataframe
    path_to_csv = path_to_data + animal + "_" + fov + "_" + experiment + "-" + run
    df_events.to_csv(path_to_csv + "_spikes.csv", index=False)
    print("\tdataframe created for: ", path_to_csv)


def load_raw_files_to_combine(animal, fov, experiment, run):
    """Load the 3 .mat files that I combine to build the .csv file with all the data."""
    path_to_data = find_path_to_data_folder(animal, fov, experiment, run)
    beh_file, spikes_file = None, None
    # load the files' names in the directory
    files = os.listdir(path_to_data)
    for file in files:
        if file.endswith("_downTrackdata.mat"):
            beh_file = loadmat(path_to_data + file)
        elif file.endswith("_spikes.mat"):
            spikes_file = loadmat(path_to_data + file)
    if (spikes_file is None) or (beh_file is None):
        print(f"Not all files found for: {animal}_{fov}_{experiment}-{run}")
    return beh_file, spikes_file


# Remove ROIs that did not pass the tests


def remove_rois_to_exclude(animal, fov, experiment, run):
    """Remove the ROIs that Ann excluded, saved on the meta file."""
    name = animal + "_" + fov + "_" + experiment + "-" + run
    path_to_csv_spikes = find_path_to_csv(animal, fov, experiment, run)
    # Get the ROIs to exclude
    rois_to_exclude = get_rois_to_exclude(animal, fov, experiment, run)
    # Load the dataframe
    df_spikes = load_csv_data(animal, fov, experiment, run)
    # Remove the ROIs
    rois_to_exclude = [str(r) for r in rois_to_exclude]
    df_spikes = df_spikes.drop(columns=rois_to_exclude)
    # Save the dataframe
    df_spikes.to_csv(path_to_csv_spikes, index=False)
    print("\tROIs: " + str(rois_to_exclude) + ", excluded from: ", name)


def get_rois_to_exclude(animal, fov, experiment, run):
    """Load the .txt with the info of all the files and the ROIs to exclude, return a list of the ROIs."""
    rois_list = []
    txt = open(path_to_rois, "r").readlines()
    for file in txt:
        # Sometimes the fov is not included in the file name
        if (
            file.split(",")[0]
            == "list_" + animal + "_" + fov + "_" + experiment + "-" + run + ".txt"
        ) or (
            file.split(",")[0]
            == "list_" + animal + "_" + experiment + "-" + run + ".txt"
        ):
            try:
                rois = file.split(",")[-1]
                # int(r)-1 because the indexes start at 1 in matlab
                rois_list = [
                    int(r) for r in rois.split("[")[1].split("]")[0].split(" ")
                ]
            except exception as e:
                print(
                    "No ROIs to exclude for: " + animal + "_" + fov + "_" + experiment + "-" + run
                )
                print(e)
        else:
            print(
                "No ROIs to exclude for: "
                + animal
                + "_"
                + fov
                + "_"
                + experiment
                + "-"
                + run
            )
    return rois_list


# Add information on movement status, angular speed and global time


def add_movements_to_csv(animal, fov, experiment, run, moving_threshold=20):
    """Add the 'movement_status' and 'angular_speed' columns to the dataframe. Add also the golbal time.
    INPUTS:
    - animal, fov, experiment, run: strings
    - moving_threshold: float, cm/s
    """
    window = 10  # number of points to average the angular speed
    moving_threshold = 20  # cm/s
    tollerance_time = 0.1  # (seconds) the time the animal can be stationary before it is considered stationary (and viceversa)
    tollerance_window = np.ceil(tollerance_time * sampling_freq)

    path_to_csv_spikes = find_path_to_csv(animal, fov, experiment, run)

    # Load the dataframe to add the 'movement_status' and 'angular_speed' columns to
    df_spikes = load_csv_data(animal, fov, experiment, run)
    # If the columns are already there, delete them
    if (
        ("angular_speed" in df_spikes.columns)
        or ("movement_status" in df_spikes.columns)
        or ("glob_time" in df_spikes.columns)
    ):
        df_spikes = df_spikes.drop(
            columns=["angular_speed", "movement_status", "glob_time"]
        )

    # Get behavioural variables
    phi = df_spikes["phi"].values
    speed = df_spikes["speed"].values
    time = df_spikes["time"].values

    # Compute angular speed
    angular_speed = compute_angular_speed(phi, 1 / sampling_freq)
    trial_starts = np.where(np.diff(time) < 0)[0]
    # Correct the angular speed for the points in which the trials are split
    for trial_start in trial_starts:
        angular_speed[trial_start] = (
            angular_speed[trial_start + 1] + angular_speed[trial_start - 1]
        ) / 2
    # apply a moving average to the angular speed
    for i in range(len(angular_speed) - window):
        angular_speed[i] = np.mean(angular_speed[i : i + window])
    # Insert a column in the dataframe after 'speed'
    df_spikes.insert(
        df_spikes.columns.get_loc("speed") + 1, "angular_speed", angular_speed
    )

    # Get a status of 'moving' vs 'stationary'
    stationary = np.abs(speed) < moving_threshold
    movement_status = np.array(["moving"] * len(speed), dtype="U10")
    movement_status[stationary] = "stationary"
    # Apply a tollerance window
    movement_status = apply_tollerance_window(movement_status, tollerance_window)
    # Insert a column in the dataframe after 'y'
    df_spikes.insert(
        df_spikes.columns.get_loc("y") + 1, "movement_status", movement_status
    )

    # Add glob_time
    glob_time = np.arange(len(time)) / sampling_freq
    df_spikes.insert(df_spikes.columns.get_loc("time") + 1, "glob_time", glob_time)

    # Save the dataframe
    df_spikes.to_csv(path_to_csv_spikes, index=False)
    return None


def compute_angular_speed(angles, delta_time):
    """
    Compute the angular speed given an array of angular positions and the time interval. Adds a 0 to the first element of the diff array to have the same length as the angles array.

    Parameters:
        angles (np.array): An array of angular positions in degrees.
        delta_time (float): The time interval between successive measurements.

    Returns:
        np.array: The angular speed in degrees per unit of time.
    """
    # Calculate the differences of every
    diff = np.diff(angles)
    # Add the first element to the differences
    diff = np.insert(diff, 0, 0)
    # Adjust differences to handle the transition from 360 to 0 degrees and vice versa
    diff = (diff + 180) % 360 - 180
    # Handle the case where the jump is from 0 to 360 negatively
    diff[diff < -180] += 360
    diff[diff > 180] -= 360
    # Calculate angular speed
    angular_speed = diff / delta_time
    return angular_speed


def apply_tollerance_window(status_list, tollerated_bins):
    """
    Apply a tollerance window to the status list. If there are statuses that are held for less than tollerated_bins, they are changed to the previous status.
    The code is split into two such that moving is prioritized over stationary.

    INPUTS:
    - status_list: list of strings (numpy)
    - tollerated_bins: int
    OUTPUTS:
    - status_list: list of strings (numpy) updated
    """
    count = count_consecutive_elements(status_list)

    # I want to correct first the points where the animal stops only briefly
    for i in np.arange(1, len(count) - 2):
        if count[i] < tollerated_bins:
            # find first index
            m = np.cumsum(count[:i])[-1] - 1
            # find last index
            n = np.cumsum(count[: (i + 2)])[-1] - 1
            if status_list[m] == "moving" and status_list[n] == "moving":
                status_list[m:n] = status_list[m]

    # I want to then correct the points where the animal doesnt actually start moving
    for i in np.arange(1, len(count) - 2):
        if count[i] < tollerated_bins:
            # find first index
            m = np.cumsum(count[:i])[-1] - 1
            # find last index
            n = np.cumsum(count[: (i + 2)])[-1] - 1
            if status_list[m] == "stationary" and status_list[n] == "stationary":
                status_list[m:n] = status_list[m]

    return status_list


def count_consecutive_elements(arr):
    """Using groupby to group consecutive identical elements and count them"""
    return [sum(1 for _ in group) for key, group in groupby(arr)]
