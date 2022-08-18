import numpy as np
import modules.features as f 

from sklearn.exceptions import NotFittedError
from sklearn.utils import check_array
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Calculates features in a segmented time series

    Adapted from: 
        https://github.com/dmbee/seglearn/tree/master/seglearn/transform.py
    """
    def __init__(self, features='default', verbose=False):
        if features == 'default':
            self.features = f.base_features()
        else:
            if not isinstance(features, dict):
                raise TypeError("Features must be of type dict")
            self.features = features 
    
        if type(verbose) != bool:
            raise TypeError("Verbose parameter expects a boolean")

        self.verbose = verbose 
        self.f_labels = None 

    def fit_transform(self, X, y):
        """
        Fit model and transform with final estimator 
        """
        Xt = self.fit(X, y).transform(X)
        idx = np.where(np.isnan(Xt))
        col_mean = np.nanmean(Xt, axis=0)
        Xt[idx] = np.take(col_mean, idx[1])
        return Xt

    def fit(self, X, y=None):
        """
        Fit the transform
        """
        # check_ts_data(X, y)
        self._reset()

        if self.verbose:
            print("X Shape: ", X.shape)

        self.f_labels = self._generate_feature_labels(X)

        return self

    def transform(self, Xt):
        """
        Transform the segmented time series data into feature data.
        If contextual data is included in X, it is returned with the feature data.

        Params: 
            X: dataframe containing time series data
        """

        self._check_if_fitted()
        check_array(Xt, dtype='numeric', ensure_2d=False, allow_nd=True)
        fts = np.column_stack([self.features[f](Xt) for f in self.features])

        return fts
    
    def _reset(self):
        self.f_labels = None

    def _check_if_fitted(self):
        if self.f_labels is None:
            raise NotFittedError("Features not fitted")

    def _check_features(self, features, Xti): 
        """ 
        Tests output of each feature against time series X 
        """
        N = Xti.shape[0]
        N_fts = len(features)
        fshapes = np.zeros((N_fts, 2), dtype=np.int)
        keys = [key for key in features]
        for i in np.arange(N_fts):
            fshapes[i] = np.row_stack(features[keys[i]](Xti)).shape

        # make sure each feature returns an array shape [N, ]
        if not np.all(fshapes[:, 0] == N):
            raise ValueError("feature function returned array with invalid length, ",
                             np.array(features.keys())[fshapes[:, 0] != N])

        return {keys[i]: fshapes[i, 1] for i in range(N_fts)}

    def _generate_feature_labels(self, Xt):
        """
        Generates string feature labels
        """
        ftr_sizes = self._check_features(self.features, Xt[0:3])
        f_labels = []

        # calculated features
        for key in ftr_sizes:
            for i in range(ftr_sizes[key]):
                f_labels += [key + '_' + str(i)]

        return f_labels