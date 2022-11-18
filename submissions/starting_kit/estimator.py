import pprint
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

from sklearn import set_config

# scikit-learn>=1.2 (which it time of writing has not been release) is needed for this.
# See: https://scikit-learn.org/dev/auto_examples/miscellaneous/plot_set_output.html#introducing-the-set-output-api
set_config(transform_output="pandas")

class RidgeWithFaultyCounterRemoved(Ridge):

    def fit(self, X, y, sample_weight=None, K=25):
        """Fit Ridge regression model.

        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Training data.

        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values.

        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # pprint.pprint(list(X.columns))
        # TODO: Process X as a dataframe and y here
        # Mind that a few estimators support missing values
        # (which is not the case of Ridge).
        return super().fit(X, y, sample_weight=sample_weight)

def _encode_dates(X):
    X = X.copy()  # modify a copy of X
    # Encode the date information from the DateOfDeparture columns
    X.loc[:, "year"] = X["date"].dt.year
    X.loc[:, "month"] = X["date"].dt.month
    X.loc[:, "day"] = X["date"].dt.day
    X.loc[:, "weekday"] = X["date"].dt.weekday
    X.loc[:, "hour"] = X["date"].dt.hour

    # Finally we can drop the original columns from the dataframe
    return X.drop(columns=["date"])


def _merge_external_data(X):
    file_path = Path(__file__).parent / "external_data.csv"
    df_ext = pd.read_csv(file_path, parse_dates=["date"])

    X = X.copy()
    # When using merge_asof left frame need to be sorted
    X["orig_index"] = np.arange(X.shape[0])
    X = pd.merge_asof(
        X.sort_values("date"), df_ext[["date", "t"]].sort_values("date"), on="date"
    )
    # Sort back to the original order
    X = X.sort_values("orig_index")
    del X["orig_index"]
    return X

def get_estimator():
    sparse_output = False
    date_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=sparse_output)
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", date_encoder, date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )

    date_encoder = FunctionTransformer(_encode_dates)

    regressor = RidgeWithFaultyCounterRemoved()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe
