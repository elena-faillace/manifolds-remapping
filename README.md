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

- `fig1`: plots regarding the experiment setup and the presentation of the problem of *remapping*.
    - :white_large_square: a. schematic of microscope and environment's setup
    - :white_large_square: b. Schematic of how two environments are different + summary of experiments structure (training + recordings)
    - :white_large_square: c. Slice of HP to shoe the aread targetd + calcium traces example
    - :white_large_square: d. Plot showing the behaviour of the mouse for 20 minutes of recordings
    - :white_large_square: e. Example of cells remapping (events over traces) over two sessions
    - :large_orange_diamond: f. Show the tuning curves ordered for one session and the another. Should generalised to all the recordings of an animal+fov together

- `fig2`: plots with main results, presentation of the model and usage to decode position across sessions.
    - :white_check_mark: a. Given an animal+fov I plotted the tuning curves embedding pre and post alignment (taking only the common neurons)
    - :white_check_mark: b. Show tuning curves before and after alignment. Specifically, shows the tuning curves without alignment (for all  common neurons); then shows the activity of all sessions projected on the neural space of the reference session (all neurons of reference session).
    - :white_check_mark: c. Show how well tuning curves from the reference session 'predict' the real tuning curves of all the other sessions (using only common neurons). Specifically, look at the correlation, the cosine similarity and the R2. 
    - :white_check_mark: d. Show how the model applied to the firing rates can predict the position of the animal across all sessions. 

- `fig3`: plots that show the robustness of the model.
    - :white_check_mark: a. Show by keeping out a neuron how its actvity is stll recovered by its relationship to the manifold. Plotted the correlation, cosine similarity and L2 distance between a tuning curve predicted by including it in the alignment or not.
    - :large_orange_diamond: b. Remove Place Cells when aligning and check how well the tuning curves can be predicted (or how well the decoder performs).
    - :white_large_square: c.
    - :white_large_square: d.

## Other files

- `tools/` contains functions to support the code for the figures
- `env.yml` all details of the python environment
- `params.py` contains variable shared across all the project
- `see_all_info_recordings.ipynb` notebook used to visualise all the information (neural+behavioural) of a given recording.

## Figures plotted so far

### Figure 2
#### Figure 2a:
![Figure 2a part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2a_1.png)
![Figure 2a part 2](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2a_2.png)
![Figure 2a part 3](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2a_3.png)

#### Figure 2b:
![Figure 2b part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2b_1.png)
![Figure 2b part 2](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2b_2.png)

#### Figure 2c:
![Figure 2c part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2c_1.png)
![Figure 2c part 2](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2c_2.png)

#### Figure 2d:
![Figure 2d part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2d_1.png)
![Figure 2d part 2](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig2/plots/PNGs/fig2d_2.png)

### Figure 3
#### Figure 3a:
![Figure 3a part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig3/plots/PNGs/fig3a_1.png)
![Figure 3a part 2](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig3/plots/PNGs/fig3a_2.png)

#### Figure 3b:
![Figure 3b part 1](https://github.com/elena-faillace/manifolds-remapping/blob/main/figures/fig3/plots/PNGs/fig3b_1.png)