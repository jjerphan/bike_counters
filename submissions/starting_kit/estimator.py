import keras
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer

BATCH_SIZE = 32
EPOCHS = 10

class ExtendedSequential(keras.Sequential):
    def fit(self, X, y):

        # TODO: Update the logic here
        return super().fit(X, y, batch_size=BATCH_SIZE, epochs=EPOCHS)

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


def get_estimator():
    date_encoder = OneHotEncoder(handle_unknown="ignore")
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", date_encoder, date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )

    date_encoder = FunctionTransformer(_encode_dates)

    # TODO: Update the logic here
    keras_regressor = ExtendedSequential()
    keras_regressor.add(keras.layers.Dense(8))
    keras_regressor.add(keras.layers.Dense(1))
    keras_regressor.compile(optimizer='sgd', loss='mse')

    pipe = make_pipeline(
        date_encoder,
        preprocessor,
        keras_regressor,
    )

    return pipe
