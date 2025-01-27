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

