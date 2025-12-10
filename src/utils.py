import pandas as pd
import numpy as np


def custom_train_test_split(
    dataframe: pd.DataFrame,
    cat_col: str,
    test_size: float = 0.3,
    random_state: int = 573,
):
    np.random.seed(random_state)

    df: pd.DataFrame = dataframe.copy()

    cat_ids: np.ndarray = df[cat_col].unique()

    random_shuffling: dict = {cat: np.random.rand() for cat in cat_ids}

    df["ordered_cat"] = df[cat_col].map(random_shuffling)

    df = df.sort_values("ordered_cat")

    cum_count = 0

    for cat, count in df.groupby(cat_col, sort=False)[cat_col].agg("count").items():
        cum_count += count
        if (cum_count / len(df)) >= (1 - test_size):
            cutoff = cat
            break

    df = df.drop("ordered_cat", axis=1).set_index(cat_col)

    train_df = df.loc[:cutoff].reset_index()
    test_df = df.loc[cutoff:].reset_index()

    return train_df, test_df


def inhouse_haversine(lat1, lon1, lat2, lon2):
    """
    Compute great-circle distance using the Haversine formula.
    Works with scalars or NumPy arrays.
    Distance is returned in kilometers.
    """

    R = 6371.0  # Earth radius in km

    # Convert to radians
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    )
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c