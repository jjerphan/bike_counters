from typing import List

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import (
    load_iris,
    load_breast_cancer,
    load_wine,
    load_digits,
    load_diabetes,
    load_linnerud,
)

from submissions.external_data.estimator import export_external_data, load_external_data


def return_data_set_as_df(sklearn_datasets_loader):
    """Utility function to load a dataset from sklearn as a as pd.DataFrame"""
    return sklearn_datasets_loader(as_frame=True).data


@pytest.mark.parametrize("random_seed", [0, 1, 2, 3, 4])
def test_concatenate(random_seed):
    """Assert that the export and loading of dataset in a single external_data.csv works

    This test is parametrised on 5 seeds to tests 5 random shuffling of
    datasets available from scikit-learn to test for robustness.
    """
    sklearn_loaders = [
        load_iris,
        load_breast_cancer,
        load_wine,
        load_digits,
        load_diabetes,
        load_linnerud,
    ]
    rng = np.random.RandomState(random_seed)

    rng.shuffle(sklearn_loaders)

    # This is functional idiomatic.
    list_of_dfs: List[pd.DataFrame] = list(map(return_data_set_as_df, sklearn_loaders))

    # Export the datasets as a concatenation in external_data.csv
    export_external_data(list_of_dfs)

    # Load the exported datasets
    list_of_exported_dfs: List[pd.DataFrame] = load_external_data()

    # Assert that all loaded exported are equivalent to the original datasets.
    for df_orig, df_export in zip(list_of_dfs, list_of_exported_dfs):
        assert (df_orig.columns == df_export.columns).all()
