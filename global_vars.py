# Path to my OneDrive folder
root_dir = '/Users/elenafaillace/Library/CloudStorage/OneDrive-ImperialCollegeLondon/hippocampus/'

# Path to the file with the ROIs to exclude
path_to_rois = root_dir + 'data/list_allexpruns.txt'

# Sampling frequency of the 2P recordings
sampling_freq = 30.9

# Chronoligical order of the experiments-runs
order_experiments = [('fam1fam2', 'fam1'),
                     ('fam1fam2', 'fam2'),
                     ('fam1fam2fam1', 'fam1'),
                     ('fam1fam2fam1', 'fam2'),
                     ('fam1fam2fam1', 'fam1r2'),
                     ('fam1nov', 'fam1'),
                     ('fam1nov', 'nov'),
                     ('fam1novfam1', 'fam1'),
                     ('fam1novfam1', 'nov'),
                     ('fam1novfam1', 'fam1r2'),
                     ('fam1fam1rev', 'fam1'),
                     ('fam1fam1rev', 'fam1rev'),
                     ('fam1fam1revfam1', 'fam1'),
                     ('fam1fam1revfam1', 'fam1rev'),
                     ('fam1fam1revfam1', 'fam1r2')]