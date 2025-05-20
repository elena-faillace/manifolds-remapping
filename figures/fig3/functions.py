# TODO: Temporary, I will later move all functioms to separate files, but not in the figures folder

import os

parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
grandparent_directory = os.path.abspath(os.path.join(parent_directory, os.pardir))
os.sys.path.append(grandparent_directory)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from scipy.stats import pearsonr
import pickle as pkl
from sklearn import linear_model  # Loss: least squares, regularization: L2
from sklearn.multioutput import MultiOutputRegressor

from tools.data_formatting import (
    get_smoothed_moving_all_data,
    get_common_indexes_n_recordings,
    smooth_tuning_curves_circularly,
    from_local_to_global_index,
)
from tools.data_manager import get_all_experiments_runs, get_fovs_given_animal
from tools.alignment import procrustes, canoncorr
from params import (
    order_experiments,
    get_colors_for_each_experiment,
    animals,
    root_dir,
    experiments_to_exclude,
)


def get_predicted_tuning_curves_one_out(all_tuning_curves, all_cells, sessions):
    """
    Keep one neuron out at the time and find the alignment and predict the neuron kept out.
    """
    ### Align the tuning curves to a reference session ###

    # Smooth the tuning curves very little for better alignment
    smoothed_tuning_curves = [
        smooth_tuning_curves_circularly(tuning_curves, 20)
        for tuning_curves in all_tuning_curves
    ]
    # Take the first session as the reference
    ref = 0
    exp0, run0 = sessions[ref]
    ref_tc = smoothed_tuning_curves[ref]
    ref_cells = all_cells[ref]

    # Keep one neuron out and align the sessions
    # Make a decoder for each neuron kept out

    predicted_tuning_curves_one_out = []
    reference_tuning_curves_one_out = []
    pca = PCA(n_components=20)
    for i in range(len(sessions) - 1):
        # Get the session to align to
        tuning_curves = smoothed_tuning_curves[i + 1]

        # Get common neurons between reference and target session
        cells_masks, ordered_cells, common_neurons = get_common_indexes_n_recordings(
            [ref_cells, all_cells[i + 1]]
        )
        # For each neuron, exclude it and then align
        tuning_curve_kept_out = []
        ref_tc_kept_out = []
        for n in range(len(common_neurons)):
            # Get indexes to remove from ref and target session
            idx_to_remove_ref = np.arange(ref_tc.shape[1])[cells_masks[0]][
                ordered_cells[0]
            ][n]
            idx_to_remove = np.arange(tuning_curves.shape[1])[cells_masks[1]][
                ordered_cells[1]
            ][n]

            # Remove the neuron from the reference and target tuning curves and get their PCA
            ref_kept_out_tc = ref_tc[:, idx_to_remove_ref]  # Use for training the decoder
            sel_tc_ref = np.delete(ref_tc, idx_to_remove_ref, axis=1)
            pca_sel_tc_ref = pca.fit_transform(sel_tc_ref)

            sel_tc = np.delete(tuning_curves, idx_to_remove, axis=1)
            pca_sel_tc = pca.fit_transform(sel_tc)

            # Align the tuning curves
            A, B, _, _, _ = canoncorr(pca_sel_tc_ref, pca_sel_tc, fullReturn=True)
            # Project the target tuning curves on the reference space
            projected_pca_tc = pca_sel_tc @ B @ np.linalg.inv(A)

            # Make a decoder from reference PC space for ref_tc
            model = linear_model.LinearRegression()
            model.fit(
                pca_sel_tc_ref, ref_kept_out_tc
            )
            pred_tc = model.predict(projected_pca_tc)

            tuning_curve_kept_out.append(pred_tc)
            ref_tc_kept_out.append(ref_kept_out_tc)

        tuning_curves_kept_out = np.array(tuning_curve_kept_out)
        ref_tc_kept_out = np.array(ref_tc_kept_out)
        predicted_tuning_curves_one_out.append(tuning_curves_kept_out.T)
        reference_tuning_curves_one_out.append(ref_tc_kept_out.T)

    return predicted_tuning_curves_one_out, reference_tuning_curves_one_out


def get_predicted_tuning_curves_common_out(all_tuning_curves, all_cells, sessions):
    """
    Keep the common neurons out and find the alignment and predict the neurons kept out.
    Rotate through the reference sessions.
    """
    ### Align the tuning curves to a reference session ##
    results = {}

    # Smooth the tuning curves very little for better alignment
    smoothed_tuning_curves = [
        smooth_tuning_curves_circularly(tuning_curves, 20)
        for tuning_curves in all_tuning_curves
    ]
    # Take one sessions at a time as the reference
    for ref in range(len(sessions) - 1):
        ref_tc = smoothed_tuning_curves[ref]
        ref_cells = all_cells[ref]
        results[sessions[ref]] = {}

        # align the sessions removing common neurons
        # Make a decoder for each neuron kept out

        predicted_tc_kept_out = []
        reference_tc_kept_out = []
        n_neurons_kept_out = []
        pca = PCA(n_components=20)
        for i in range(len(sessions) - 1):
            if i != ref:
                # Get the session to align to
                tuning_curves = smoothed_tuning_curves[i]

                # Get common neurons between reference and target session
                cells_masks, ordered_cells, common_neurons = get_common_indexes_n_recordings(
                    [ref_cells, all_cells[i]]
                )
                
                if len(common_neurons) < tuning_curves.shape[1]:
                    print(f"Aligning {sessions[ref]} and {sessions[i]} with {len(common_neurons)} common neurons")
                    # Get indexes to remove from ref and target session
                    idx_to_remove_ref = np.arange(ref_tc.shape[1])[cells_masks[0]][
                        ordered_cells[0]
                    ]
                    idx_to_remove = np.arange(tuning_curves.shape[1])[cells_masks[1]][
                        ordered_cells[1]
                    ]

                    # Remove the neuron from the reference and target tuning curves and get their PCA
                    ref_kept_out_tc = ref_tc[:, idx_to_remove_ref]  # Use for training the decoder
                    sel_tc_ref = np.delete(ref_tc, idx_to_remove_ref, axis=1)
                    pca_sel_tc_ref = pca.fit_transform(sel_tc_ref)

                    sel_tc = np.delete(tuning_curves, idx_to_remove, axis=1)
                    pca_sel_tc = pca.fit_transform(sel_tc)

                    # Align the tuning curves
                    A, B, _, _, _ = canoncorr(pca_sel_tc_ref, pca_sel_tc, fullReturn=True)
                    # Project the target tuning curves on the reference space
                    projected_pca_tc = pca_sel_tc @ B @ np.linalg.inv(A)

                    # Make a decoder from reference PC space for ref_tc
                    model = MultiOutputRegressor(linear_model.LinearRegression())
                    model.fit(
                        pca_sel_tc_ref, ref_kept_out_tc
                    )
                    pred_tc = model.predict(projected_pca_tc)

                    predicted_tc_kept_out.append(pred_tc)
                    reference_tc_kept_out.append(ref_kept_out_tc)
                    n_neurons_kept_out.append(len(common_neurons))
            
        # Save the results
        results[sessions[ref]]["predicted_tc"] = predicted_tc_kept_out
        results[sessions[ref]]["reference_tc"] = reference_tc_kept_out
        results[sessions[ref]]["n_neurons_kept_out"] = n_neurons_kept_out
    return results