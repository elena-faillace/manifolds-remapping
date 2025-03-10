# manifolds-remapping
Repository for preparing paper on representational drift (remapping) in CA1

## Prepare data from original files (.mat)

Use prepare_data.ipynb to use the files available from Ann's recording: 
- mouse_session-env_downTrackdata.mat contains the information about the behaviour
- mouse_session-env_ref_date_PFmap_output.mat contains the information about the place cells
- mouse_session-env_ref_date_segment_output.mat contains the information about the df/f
- mouse_session-env_spikes.mat contains the information about the spikes

`prepare_data.ipynb` will:
1. combine the data in `_downTrackdata.mat` with the spikes in `_spikes.mat`, to create a .csv file `_spikes.csv`.
2. add to the .csv file the information on *angular_speed*, *global_time* and *movement_status*.

## Figures

- `fig1.ipynb`: plots regarding the experiment setup and the presentation of the problem of *remapping*.
    - :white_large_square: a. schematic of microscope and environment's setup
    - :white_large_square: b. Schematic of how two environments are different + summary of experiments structure (training + recordings)
    - :white_large_square: c. Slice of HP to shoe the aread targetd + calcium traces example
    - :white_large_square: d. Plot showing the behaviour of the mouse for 20 minutes of recordings
    - :white_large_square: e. Example of cells remapping (events over traces) over two sessions
    - :large_orange_diamond: f. Show the tuning curves ordered for one session and the another. Should generalised to all the recordings of an animal+fov together

- `fig2.ipynb`: plots with main results, presentation of the model and usage to decode position across sessions.
    - :white_check_mark: a. Given an animal+fov I plotted the tuning curves embedding pre and post alignment (taking only the common neurons)
        - TODO: need to update the analysis with registred neurons across days. Could make multiple plots for within day aligment and across days
    - :white_large_square: b. 
    - :white_large_square: c.
    - :white_large_square: d.

## Other files

- `tools/` contains functions to support the code for the figures
- `env.yml` all details of the python environment
- `global_vars.py` contains variable shared across all the project
- `see_all_info_recordings.ipynb` notebook used to visualise all the information (neural+behavioural) of a given recording.
