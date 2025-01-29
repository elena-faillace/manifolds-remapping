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

colors_each_experiment = {('fam1fam2','fam1'): '#e85d04',
                          ('fam1fam2fam1','fam1'): '#e85d04',
                          ('fam1fam2', 'fam2'): '#ff8800',
                          ('fam1fam2fam1', 'fam2'): '#ff8800',
                          ('fam1fam2fam1', 'fam1r2'): '#ffba08',
                          ('fam1nov', 'fam1'): '#208b3a',
                          ('fam1novfam1', 'fam1'): '#208b3a',
                          ('fam1nov', 'nov'): '#99ca3c',
                          ('fam1novfam1', 'nov'): '#99ca3c',
                          ('fam1novfam1', 'fam1r2'): '#cbdb47',
                          ('fam1fam1rev', 'fam1'): '#7f25fb',
                          ('fam1fam1revfam1', 'fam1'): '#7f25fb',
                          ('fam1fam1rev', 'fam1rev'): '#d727fc',
                          ('fam1fam1revfam1', 'fam1rev'): '#d727fc',
                          ('fam1fam1revfam1', 'fam1r2'): '#fd23de'}

def get_colors_for_each_experiment(sessions):
    """Given a list of tuples (experiments-runs) returns a list of colors."""
    return [colors_each_experiment[session] for session in sessions]
