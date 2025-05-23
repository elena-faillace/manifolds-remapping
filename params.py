# Path to my OneDrive folder
root_dir = "/Users/elenafaillace/Library/CloudStorage/OneDrive-ImperialCollegeLondon/hippocampus/"

# Path to the file with the ROIs to exclude
path_to_rois = root_dir + "data/list_allexpruns.txt"

# Sampling frequency of the 2P recordings
sampling_freq = 30.9

# Chronoligical order of the experiments-runs
order_experiments = [
    ("fam1fam2", "fam1"),
    ("fam1fam2", "fam2"),
    ("fam1fam2fam1", "fam1"),
    ("fam1fam2fam1", "fam2"),
    ("fam1fam2fam1", "fam1r2"),
    ("fam1fam2s2", "fam1"),
    ("fam1fam2s2", "fam2"),
    ("fam1fam2s3", "fam1"),
    ("fam1fam2s3", "fam2"),
    ("fam1nov", "fam1"),
    ("fam1nov", "nov"),
    ("fam1novfam1", "fam1"),
    ("fam1novfam1", "nov"),
    ("fam1novfam1", "fam1r2"),
    ("fam1fam1rev", "fam1"),
    ("fam1fam1rev", "fam1rev"),
    ("fam1fam1revfam1", "fam1"),
    ("fam1fam1revfam1", "fam1rev"),
    ("fam1fam1revfam1", "fam1r2"),
]

# For consistency, the colors of the experiments are the same across all the plots
colors_each_experiment = {
    ("fam1fam2", "fam1"): "#e85d04",
    ("fam1fam2fam1", "fam1"): "#e85d04",
    ("fam1fam2", "fam2"): "#ff8800",
    ("fam1fam2fam1", "fam2"): "#ff8800",
    ("fam1fam2fam1", "fam1r2"): "#ffba08",
    ("fam1fam2s2", "fam1"): "#0d41e1",
    ("fam1fam2s2", "fam2"): "#0a85ed",
    ('fam1fam2s3', 'fam1'): '#0d41e1',
    ('fam1fam2s3', 'fam2'): '#0a85ed',
    ("fam1nov", "fam1"): "#208b3a",
    ("fam1novfam1", "fam1"): "#208b3a",
    ("fam1nov", "nov"): "#99ca3c",
    ("fam1novfam1", "nov"): "#99ca3c",
    ("fam1novfam1", "fam1r2"): "#cbdb47",
    ("fam1fam1rev", "fam1"): "#7f25fb",
    ("fam1fam1revfam1", "fam1"): "#7f25fb",
    ("fam1fam1rev", "fam1rev"): "#d727fc",
    ("fam1fam1revfam1", "fam1rev"): "#d727fc",
    ("fam1fam1revfam1", "fam1r2"): "#fd23de",
}


def get_colors_for_each_experiment(sessions):
    """Given a list of tuples (experiments-runs) returns a list of colors."""
    return [colors_each_experiment[session] for session in sessions]


# List of animals
animals = [
    "m62",
    "m66",
    "m70",
    "m116",
    "m117",
    "m120",
    "m127",
    "m129",
    "m130",
    "m134",
    "m135",
    "m111",
    "m118",
    "m125",
    "m139",
    "m140",
    "m141",
    "m77",
    "m79",
    "m121",
    "m128",
    "m132",
]

# List of experiments to exclude
experiments_to_exclude = [
    ("m66", "fov1", "fam1fam2", "fam1"),
    ("m117", "fov2", "fam1fam2", "fam1"),
    ("m117", "fov2", "fam1fam2", "fam2"),
]

# Animal types AD or WT
animal_types = {
    "m111": "5xFAD",
    "m118": "5xFAD",
    "m125": "5xFAD",
    "m139": "5xFAD",
    "m140": "5xFAD",
    "m141": "5xFAD",
    "m77": "5xFAD",
    "m79": "5xFAD",
    "m121": "5xFAD",
    "m128": "5xFAD",
    "m132": "5xFAD",
    "m62": "WT",
    "m66": "WT",
    "m70": "WT",
    "m116": "WT",
    "m117": "WT",
    "m120": "WT",
    "m127": "WT",
    "m129": "WT",
    "m130": "WT",
    "m134": "WT",
    "m135": "WT",
}

animals_age = {
    'm111': 'old',
    'm118': 'old',
    'm125': 'old',
    'm139': 'old',
    'm140': 'old',
    'm141': 'old',
    'm77': 'young',
    'm79': 'young',
    'm121': 'young',
    'm128': 'young',
    'm132': 'young',
    'm62': 'old',
    'm66': 'old',
    'm70': 'old',
    'm116': 'old',
    'm117': 'old',
    'm120': 'young',
    'm127': 'young',
    'm129': 'young',
    'm130': 'young',
    'm134': 'young',
    'm135': 'young',
}

# Colors for plotting different results
results_colors = {"MDA": "#14ddac", "CMSE": "#06dcd5"}
