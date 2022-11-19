from pathlib import Path

import numpy as np
import pandas as pd

from typing import List
from pathlib import Path

from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge

# Must be adapted accordingly to your project layout
PARENT_FOLDER_PATH = Path(__file__).parent

EXTERNAL_DATA_CSV_PATH = PARENT_FOLDER_PATH / "external_data.csv"

TOP_HEADER_SEPARATOR = ","


def export_external_data(list_of_dfs: List[pd.DataFrame]):
    """
    Export the concatenation of several pandas.DataFrame
    with their index and header in `external_data.csv`.

    An header is prepended to list the length (number of rows)
    of each pandas.DataFrame in order.
    Parameters
    ----------
    list_of_dfs: List[pd.DataFrame]
        The datasets to export in the concatenated CSV format.
    """

    dfs_n_samples: List[str] = map(str, map(len, list_of_dfs))

    with open(EXTERNAL_DATA_CSV_PATH, mode="w") as csv_file:
        # We store the number of samples per datasets in a top header to be able to
        # load them then.
        n_samples_header: str = f"{TOP_HEADER_SEPARATOR.join(dfs_n_samples)}\n"
        csv_file.write(n_samples_header)

    for df in list_of_dfs:
        # Append lines in order, with their index and their header
        df.to_csv(EXTERNAL_DATA_CSV_PATH, mode="a", index=True, header=True)


def load_external_data():
    """
    Export the concatenation of several pandas.DataFrame
    with their index and header in `external_data.csv`.

    A header is prepended to list the length (number of rows)
    of each pandas.DataFrame in order.
    Parameters
    ----------
    list_of_dfs: List[pd.DataFrame]
        The datasets to load from the concatenated CSV format.
    """

    with open(EXTERNAL_DATA_CSV_PATH, mode="r") as csv_file:
        n_samples_header: str = csv_file.readline()

    # We load the number of samples per datasets in a top header
    # to be able to compute offsets of lines to skip for each dataset
    # The extra [1] is used to skip the top header.
    offsets = np.array(
        [1] + list(map(int, n_samples_header.split(TOP_HEADER_SEPARATOR)))
    )

    dfs_lines_to_skip = np.cumsum(offsets)

    dfs = []

    for n_dataset_passed in range(len(offsets) - 1):
        # We add n_dataset_passed to skip lines of previous datasets' headers.
        n_rows_to_skip = dfs_lines_to_skip[n_dataset_passed] + n_dataset_passed
        df_last_row = dfs_lines_to_skip[n_dataset_passed + 1] + n_dataset_passed
        df_n_rows = df_last_row - n_rows_to_skip

        # Load datasets with
        df = pd.read_csv(
            EXTERNAL_DATA_CSV_PATH,
            index_col=0,
            # Header of the dataset are one its first line.
            header=0,
            skiprows=n_rows_to_skip,
            nrows=df_n_rows,
        )

        dfs.append(df)

    return dfs


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
    df_ext = pd.read_csv(EXTERNAL_DATA_CSV_PATH, parse_dates=["date"])

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
    date_encoder = FunctionTransformer(_encode_dates)
    date_cols = ["year", "month", "day", "weekday", "hour"]

    categorical_encoder = OneHotEncoder(handle_unknown="ignore")
    categorical_cols = ["counter_name", "site_name"]

    preprocessor = ColumnTransformer(
        [
            ("date", OneHotEncoder(handle_unknown="ignore"), date_cols),
            ("cat", categorical_encoder, categorical_cols),
        ]
    )
    regressor = Ridge()

    pipe = make_pipeline(
        FunctionTransformer(_merge_external_data, validate=False),
        date_encoder,
        preprocessor,
        regressor,
    )

    return pipe


if __name__ == "__main__":
    dfs = load_external_data()
