import numpy as np
from sklearn.model_selection import KFold
from sklearn import linear_model


### DECODERS ###

def make_k_decoders(X, phi, k=4, full_return=False):
    """Make k decoders by randomly splitting the data into training and testing sets.
    Returns the decoders and the metrics in the test dataset. 
    INPUTS:
    - X: firing rates data
    - phi: the circular variable
    - k: number of decoders to make
    - full_return=True, returns the predicted and real angles for each decoder.
    OUTPUTS:
    - decoders: list of decoders
    - CMSEs: list of circular mean squared errors
    - MDAs: list of mean directional accuracies
    - (optional) predicted_angles, real_angles
    """
    # Parameters of the decoder
    bins_before, bins_current, bins_after = 0, 1, 0

    # Get the target variables from phi
    sin_phi = np.sin(phi * np.pi / 180)
    cos_phi = np.cos(phi * np.pi / 180)

    # Split the data into k decoders
    decoders = []
    CMSEs = []
    MDAs = []
    if full_return:
        predicted_angles = []
        real_angles = []
    kf = KFold(n_splits=k, shuffle=True) # TODO: can only do this if I am taking one bin at the time, otherwise history doesn-t make sense
    for train_index, test_index in kf.split(X):

        # Get the training and testing data
        X_train, Y_train = X[train_index,:], [sin_phi[train_index], cos_phi[train_index]]
        X_test, Y_test = X[test_index,:], [sin_phi[test_index], cos_phi[test_index]]

        # Put them in the right format
        ## Get the history
        X_train_ = get_spikes_with_history(X_train, bins_before, bins_current, bins_after)
        X_test_ = get_spikes_with_history(X_test, bins_before, bins_current, bins_after)
        ## Remove nans
        nan_mask = np.isnan(X_train_).any(axis=(1,2))
        X_train_ = X_train_[~nan_mask,:,:]
        Y_train = [Y_train[0][~nan_mask], Y_train[1][~nan_mask]]
        nan_mask = np.isnan(X_test_).any(axis=(1,2))
        X_test_ = X_test_[~nan_mask,:,:]
        Y_test = [Y_test[0][~nan_mask], Y_test[1][~nan_mask]]
        ## Reshape
        X_train_flat = X_train_.reshape(X_train_.shape[0], X_train_.shape[1]*X_train_.shape[2])
        Y_train = np.array(Y_train).T
        X_test_flat = X_test_.reshape(X_test_.shape[0], X_test_.shape[1]*X_test_.shape[2])
        Y_test = np.array(Y_test).T

        # Fit the model
        decoder = WienerFilterRegression(regularization='l2')
        decoder.fit(X_train_flat, Y_train)
        decoders.append(decoder)

        # Predict and get metrics
        Y_predicted = decoder.predict(X_test_flat)
        sin_predicted, cos_predicted = Y_predicted[:,0], Y_predicted[:,1]
        sin_true, cos_true = Y_test[:,0], Y_test[:,1]
        CMSEs.append(circular_mean_squared_error(sin_true, sin_predicted, cos_true, cos_predicted))
        phi_predicted = np.arctan2(sin_predicted, cos_predicted) * 180 / np.pi
        phi_real = np.arctan2(sin_true, cos_true) * 180 / np.pi
        MDAs.append(mean_directional_accuracy(phi_real, phi_predicted))
        if full_return:
            predicted_angles.append(phi_predicted)
            real_angles.append(phi_real)
    
    if full_return:
        return decoders, CMSEs, MDAs, predicted_angles, real_angles
    return decoders, CMSEs, MDAs

def apply_decoders(decoders, X, phi):
    """Given a aset of decoders and the data, apply the decoders and return the metrics and predicted angles.
    """
    bins_before, bins_current, bins_after = 0, 1, 0
    # Format the data
    Y = [np.sin(phi * np.pi / 180), np.cos(phi * np.pi / 180)]
    X_ = get_spikes_with_history(X, bins_before, bins_current, bins_after)
    nan_mask = np.isnan(X_).any(axis=(1,2))
    X_ = X_[~nan_mask,:,:]
    Y = [Y[0][~nan_mask], Y[1][~nan_mask]]
    X_flat = X_.reshape(X_.shape[0], X_.shape[1]*X_.shape[2])
    Y = np.array(Y).T
    sin_true, cos_true = Y[:,0], Y[:,1]
    phi_true = np.arctan2(sin_true, cos_true) * 180 / np.pi
    # Apply the decoders
    predicted_angles = []
    real_angles = []
    CMSEs = []
    MDAs = []
    for decoder in decoders:
        Y_predicted = decoder.predict(X_flat)
        sin_predicted, cos_predicted = Y_predicted[:,0], Y_predicted[:,1]
        CMSEs.append(circular_mean_squared_error(sin_true, sin_predicted, cos_true, cos_predicted))
        phi_predicted = np.arctan2(sin_predicted, cos_predicted) * 180 / np.pi
        predicted_angles.append(phi_predicted)
        real_angles.append(phi_true)
        MDAs.append(mean_directional_accuracy(phi_true, phi_predicted))

    return CMSEs, MDAs, predicted_angles, real_angles



### DECODER PERFORMANCE METRICS ###

def mean_directional_accuracy(angle_true, angle_predicted):
    """Given a circular variable, compute the mean directionatl accuracy (MDA). Results are in [-1,+1].
    INPUTS:
    - angle_true, angle_predicted: arrays of angles in degrees (0,360)
    OUTPUT:
    - mda: mean directional accuracy
    """
    angle_diff = angle_predicted - angle_true
    mda = np.mean(np.cos(angle_diff * np.pi / 180))
    return mda

def circular_mean_squared_error(sin_true, sin_predicted, cos_true, cos_predicted):
    """Given a circular variable, compute the circular mean squared error (CMSE).
    INPUTS:
    - sin_true, sin_predicted, cos_true, cos_predicted: arrays of sin and cos values
    OUTPUT:
    - cmse: circular mean squared error
    """
    return np.mean(((sin_true - sin_predicted)**2 + (cos_true - cos_predicted)**2))


##### GET_SPIKES_WITH_HISTORY #####

def get_spikes_with_history(neural_data,bins_before,bins_after,bins_current=1):
    """
    Function that creates the covariate matrix of neural activity

    Parameters
    ----------
    neural_data: a matrix of size "number of time bins" x "number of neurons"
        the number of spikes in each time bin for each neuron
    bins_before: integer
        How many bins of neural data prior to the output are used for decoding
    bins_after: integer
        How many bins of neural data after the output are used for decoding
    bins_current: 0 or 1, optional, default=1
        Whether to use the concurrent time bin of neural data for decoding

    Returns
    -------
    X: a matrix of size "number of total time bins" x "number of surrounding time bins used for prediction" x "number of neurons"
        For every time bin, there are the firing rates of all neurons from the specified number of time bins before (and after)
    """

    num_examples=neural_data.shape[0] #Number of total time bins we have neural data for
    num_neurons=neural_data.shape[1] #Number of neurons
    surrounding_bins=bins_before+bins_after+bins_current #Number of surrounding time bins used for prediction
    X=np.empty([num_examples,surrounding_bins,num_neurons]) #Initialize covariate matrix with NaNs
    X[:] = np.nan
    #Loop through each time bin, and collect the spikes occurring in surrounding time bins
    #Note that the first "bins_before" and last "bins_after" rows of X will remain filled with NaNs, since they don't get filled in below.
    #This is because, for example, we cannot collect 10 time bins of spikes before time bin 8
    start_idx=0
    for i in range(num_examples-bins_before-bins_after): #The first bins_before and last bins_after bins don't get filled in
        end_idx=start_idx+surrounding_bins; #The bins of neural data we will be including are between start_idx and end_idx (which will have length "surrounding_bins")
        X[i+bins_before,:,:]=neural_data[start_idx:end_idx,:] #Put neural data from surrounding bins in X, starting at row "bins_before"
        start_idx=start_idx+1;
    return X


##################### WIENER FILTER ##########################

class WienerFilterRegression(object):
    """
    Class for the Wiener Filter Decoder

    There are no parameters to set.

    This simply leverages the scikit-learn linear regression.
    """

    def __init__(self, regularization=None):
        if regularization == None:
            self.model = (
                linear_model.LinearRegression()
            )  # Initialize linear regression model
        elif regularization == "l2":
            self.model = linear_model.Ridge()
        elif regularization == "LARS":
            self.model = linear_model.Lars()
        else:
            print(
                "Error, not right regularization inserted, initialising normal linear regression"
            )
            self.model = linear_model.LinearRegression()
        
        return

    def fit(self, X_flat_train, y_train):
        """
        Train Wiener Filter Decoder

        Parameters
        ----------
        X_flat_train: numpy 2d array of shape [n_samples,n_features]
            This is the neural data.
            See example file for an example of how to format the neural data correctly

        y_train: numpy 2d array of shape [n_samples, n_outputs]
            This is the outputs that are being predicted
        """
        self.model.fit(X_flat_train, y_train)  # Train the model

    def predict(self, X_flat_test):
        """
        Predict outcomes using trained Wiener Cascade Decoder

        Parameters
        ----------
        X_flat_test: numpy 2d array of shape [n_samples,n_features]
            This is the neural data being used to predict outputs.

        Returns
        -------
        y_test_predicted: numpy 2d array of shape [n_samples,n_outputs]
            The predicted outputs
        """

        y_test_predicted = self.model.predict(X_flat_test)  # Make predictions
        return y_test_predicted
