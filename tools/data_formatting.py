"""Functions to prepare the data for future analysis."""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt
from scipy.signal import butter, filtfilt

from tools.data_manager import load_csv_data, load_ca_data


def lowpass_filter(data, cutoff=1, fs=30.9, order=4):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyquist  # Normalized cutoff
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # Butterworth filter
    return filtfilt(b, a, data)  # Apply zero-phase filtering

# TODO: maybe delete this later if calcium traces are not used
def get_smoothed_moving_ca(animal, fov, experiment, run):
    """Implement the pre-processing as Ann has suggested for the calcium traces."""
    # Load the calcium traces
    ca_df = load_ca_data(animal, fov, experiment, run)
    # Select only the columns with the traces (they contain a number)
    neurons_ids = [col for col in ca_df.columns if col.isdigit()]
    ca = ca_df[neurons_ids].values
    # Get baseline value and remove it
    threshold = np.percentile(ca, 35)
    ca = ca - threshold
    # Smooth the traces but using a median filter with window of 7
    ca = medfilt(ca, (7, 1))
    # Lowpass filter
    ca = lowpass_filter(ca, cutoff=1, fs=30.9)

    # Load the spikes to get the moving information
    spikes_df = load_csv_data(animal, fov, experiment, run)
    time = ca_df['time'].values
    phi = ca_df['phi'].values
    moving_masks = spikes_df['movement_status'].values=='moving'
    cells = ca_df.columns[ca_df.columns.str.contains(r'^\d', regex=True)]
    
    # TODO: should control for NaN
    if np.isnan(ca).any().sum() > 0:
        print("Warning: Some calcium traces are NaN")
    # Remove the stationary points
    ca = ca[moving_masks,:]
    time = time[moving_masks]
    phi = phi[moving_masks]
    return ca, time, phi, cells

def get_smoothed_moving_spikes(animal, fov, experiment, run, bins_compress=3, sigma_smoothing=3, portion_to_remove=0.0):
    """Function that pre-process the spikes from the experiments such that they are ready for further analysis. 
    Need to test the parameters first in another notebook to make sure they are adequate to those recordings.
    This function:
    * Removes a first portion of the recordings.
    * Bins the spikes.
    * Convolutes the spikes with a gaussian kernel.
    * Square roots the firing rates.
    * Removes the stationary points.
    INPUTS: 
    - animal,fov,experiment,run: list of dictionaries with the information of the experiments to be processed. 
    - bins_compress: integer with the number of bins to compress the data.
    - sigma_smoothing: float with the sigma for the gaussian kernel to smooth the data.
    - portion_to_remove: float with the portion of the data to remove from the beginning of the recordings. 
    OUTPUTS:
    - sel_firing_rates: list of dataframes with the firing rates of the experiments pre-processed, the columns are the cells idxs.
    - sel_time: list of numpy arrays with the time of the experiments pre-processed.
    - sel_phi: list of numpy arrays with the phi of the experiments pre-processed.
    - cells: list of strings with the idxs of the cells.
    """
    # Load the data
    df_orig = load_csv_data(animal, fov, experiment, run)
    time = df_orig['time'].values
    phi = df_orig['phi'].values
    moving_masks = df_orig['movement_status'].values=='moving'
    cells = df_orig.columns[df_orig.columns.str.contains(r'^\d', regex=True)]
    spikes = df_orig[cells].values
    # Remove the first portion of the recordings
    spikes = spikes[int(len(spikes)*portion_to_remove):,:]
    time = time[int(len(time)*portion_to_remove):]
    phi = phi[int(len(phi)*portion_to_remove):]
    moving_masks = moving_masks[int(len(moving_masks)*portion_to_remove):]
    # If there are nans in the spikes, remove the neurons and time points (choose the right order)
    while np.isnan(spikes).any().sum() > 0:
        if np.isnan(spikes).any(axis=0).sum() == spikes.shape[1]:
            # Remove time-points first
            print(f'WARNING: Removing time-points with NaN values from {animal}_{fov}_{experiment}-{run}: from {spikes.shape[0]} to {np.sum(~np.isnan(spikes).any(axis=1))}')
            time = time[~np.isnan(spikes).any(axis=1)]
            phi = phi[~np.isnan(spikes).any(axis=1)]
            moving_masks = moving_masks[~np.isnan(spikes).any(axis=1)]
            spikes = spikes[~np.isnan(spikes).any(axis=1),:]
        if np.isnan(spikes).any(axis=1).sum() == spikes.shape[0]:
            # Remove neurons first
            print(f'WARNING: Removing neurons with NaN values from {animal}_{fov}_{experiment}-{run}: from {spikes.shape[1]} to {np.sum(~np.isnan(spikes).any(axis=0))}')
            cells = cells[~np.isnan(spikes).any(axis=0)]
            spikes = spikes[:,~np.isnan(spikes).any(axis=0)]
    # Bin the dataset
    n_bins = spikes.shape[0]//bins_compress
    spikes_b = np.sum(spikes[:n_bins*bins_compress].reshape(-1, bins_compress, spikes.shape[1]), axis=1)
    time_b = np.mean(time[:n_bins*bins_compress].reshape(-1, bins_compress), axis=1)
    phi_b = np.mean(phi[:n_bins*bins_compress].reshape(-1, bins_compress), axis=1)
    moving_masks_b = np.sum(moving_masks[:n_bins*bins_compress].reshape(-1, bins_compress), axis=1)>0
    # Convolute with a Gaussian kernel
    firing_rates = get_firing_rates(spikes_b.T,sigma=sigma_smoothing).T
    # Square root the firing rates
    firing_rates = np.sqrt(firing_rates)
    # Remove the stationary points
    sel_firing_rates = firing_rates[moving_masks_b,:]
    sel_time = time_b[moving_masks_b]
    sel_phi = phi_b[moving_masks_b]
    return sel_firing_rates, sel_time, sel_phi, cells

def get_firing_rates(events, sigma):
    """Return firing rates from events trains. It might return less neurons if NaN values are present.
    INPUTS:
    - events = events trains of the neurons (neurons x timepoints)
    - sigma = sigma of the gaussian filter (default: 6.2 for 30.9Hz from literature)
    """

    # Remove rows (neurons) with NaN values
    #sel_eve = events[~(np.sum(np.isnan(events), axis=1)>0)]
    # Apply gaussian filter along one axis
    frates = gaussian_filter1d(events, sigma=sigma, axis=1)
    return frates

def get_smoothed_moving_all_data(animal, fov, experiment, run, n_points=360, portion_to_remove=0.0):
    """Load the data that has been binned and smoothed.
    INPUTS:
    - animal, fov, experiment, run: strings, names of the data to load
    - n_components: number of components to keep from the PCA of the average manifold ring (if -1 use all)
    - n_points: number of points to use for the average manifold ring
    OUTPUTS:
    firing_rates, time, phi, cells, average_firing_rates, phi_bins
    """
    # Load the data binned and smoothed
    firing_rates, time, phi, cells = get_smoothed_moving_spikes(animal, fov, experiment, run, portion_to_remove=portion_to_remove)
    # Remove time-points where phi is nan
    firing_rates = firing_rates[~np.isnan(phi)]
    time = time[~np.isnan(phi)]
    phi = phi[~np.isnan(phi)]
    # Load the tuning curves
    average_firing_rates, phi_bins = get_tuning_curves(firing_rates, phi, n_points=n_points)
    # Unwrap the time
    tdiff = np.diff(time)
    time_unwrapped = np.zeros(len(time))
    time_unwrapped[0] = time[0]
    for i in range(1, len(time_unwrapped)):
        if tdiff[i-1] > 0:
            time_unwrapped[i] = time_unwrapped[i-1] + tdiff[i-1]
        else:
            time_unwrapped[i] = time_unwrapped[i-1] + time[i]
    return firing_rates, time_unwrapped, phi, cells, average_firing_rates, phi_bins

def get_tuning_curves(firing_rates, phi, n_points):
    """Find the tunign curves manifold given the numnber of bins to keep.
    INPUTS: 
    - firing_rates: 2D array of shape (time-samples, neurons)
    - phi: 1D array of shape (time-samples)
    - n_points: number of points in the ring
    OUTPUTS:
    - ring_neural: 2D array of shape (n_points, neurons)
    """
    if np.isnan(phi).any().sum() > 0:
        print("Warning: Some angles are NaN")

    # To be sure the angles are within 360
    phi_mod = phi % 360
    dphi = 360/n_points
    bin_idx = np.floor(phi_mod / dphi).astype(int)
    # To be used to store the average firing rates and track the number of samples in each bin
    ring_neural = np.zeros((n_points, firing_rates.shape[1]))
    counts = np.zeros(n_points, dtype=int)
    for i in range(len(phi_mod)):
        ring_neural[bin_idx[i], :] += firing_rates[i]
        counts[bin_idx[i]] += 1
    for b in range(n_points):
        if counts[b] > 0:
            ring_neural[b, :] /= counts[b]
        else:
            ring_neural[b, :] = 0
    # Define angles associated to each bin
    points_phi = np.arange(n_points) * dphi

    if np.isnan(ring_neural).any():
        print("Warning: Some bins are empty; returning NaN for those bins.")
        for b in range(ring_neural.shape[0]):
            if np.isnan(ring_neural[b, :]).any():
                # Deal with nans at the beginning and ending of the ring
                if b == 0:
                    ring_neural[b, :] = (ring_neural[b+1, :] + ring_neural[-1, :])/2
                elif b == ring_neural.shape[0]-1:
                    ring_neural[b, :] = (ring_neural[b-1, :] + ring_neural[0, :])/2
                else:
                    ring_neural[b, :] = (ring_neural[b-1, :] + ring_neural[b+1, :])/2

    return ring_neural, points_phi

# TODO: delete this
def get_common_indexes_2recordings(cells_run1, cells_run2):
    """
    Given two lists with the cells indexes find a common order.
    Return the cells in common and the order they need to be selected. 
    First remove the not common cells and then order them.
    TODO: should generalise to more than 2 recordings.
    OUTPUTS:
    - sel_cells_run1: bool array for the cells in run1 to keep
    - sel_cells_run2: bool array for the cells in run2 to keep
    - ordered_cells_run1: ordered indexes for run1
    - ordered_cells_run2: ordered indexes for run2
    """
    # Select only common cells
    common_cells = np.intersect1d(cells_run1, cells_run2)
    c_cells_run1_mask = np.isin(cells_run1, common_cells)
    c_cells_run2_mask = np.isin(cells_run2, common_cells)
    c_cells_run1 = cells_run1[c_cells_run1_mask]
    c_cells_run2 = cells_run2[c_cells_run2_mask]
    # Order the cells
    ordered_cells_run1 = np.argsort([int(c) for c in c_cells_run1])
    ordered_cells_run2 = np.argsort([int(c) for c in c_cells_run2])

    return c_cells_run1_mask, c_cells_run2_mask, ordered_cells_run1, ordered_cells_run2

def get_common_indexes_n_recordings(cells_list):
    """
    Given a list of list of cells indexes find a common order.
    Return the cells in common and the order they need to be selected. 
    First remove the not-common cells and then order them.
    OUTPUTS:
    - sel_cells_list: list of bool arrays for the cells in the list to keep
    - ordered_cells_list: list of ordered indexes for the list
    """
    # Select common cells
    common_cells = list(set(cells_list[0]).intersection(*cells_list[1:]))
    sel_cells_list = []
    ordered_cells_list = []
    for cells in cells_list:
        c_cells_mask = np.isin(cells, common_cells)
        c_cells = cells[c_cells_mask]
        # Order the cells
        ordered_cells = np.argsort([int(c) for c in c_cells])
        sel_cells_list.append(c_cells_mask)
        ordered_cells_list.append(ordered_cells)
    return sel_cells_list, ordered_cells_list

def smooth_tuning_curves_circularly(tuning_curves, kernel_size):
    """
    Given an array of tuning curves smooth them circularly. 
    Use moving average such that the beginning and ending of the array are connected.
    INPUTS:
    - tuning_curves: 2D array of shape (n_points, n_neurons)
    - smooth_kernel: int with the size of the kernel to smooth
    OUTPUTS:
    - smoothed_tuning_curves: 2D array of shape (n_points, n_neurons)
    """
    kernel = np.ones((kernel_size,))/kernel_size
    pad_width = len(kernel) - 1
    smoothed_tuning_curves = []
    for i in range(tuning_curves.shape[1]):
        padded_array = np.pad(tuning_curves[:, i], pad_width=((pad_width,),), mode='wrap')
        smoothed_tuning_curves.append(np.convolve(padded_array, kernel, mode='valid')[:len(tuning_curves[:,i])])
    return np.array(smoothed_tuning_curves).T
