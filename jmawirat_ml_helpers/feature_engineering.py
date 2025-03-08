import pandas as pd
import numpy as np

def add_lag_features(df, col_lag_map, sort_col=None):
    """
    Creates lagged features for specified columns in a DataFrame,
    ensuring data is sorted (by index or a given column) before creating the lags.
    Afterward, it reverts the DataFrame to the original row ordering.

    - If sort_col is None, the DataFrame is sorted by its index.
    - If sort_col is a column name, the DataFrame is sorted by that column.
    - If sort_col matches the named index (df.index.name), the DataFrame is sorted by index.
      (If the index is unnamed, this match won't happen, so you should pass sort_col=None
       to indicate you want to sort by the index.)

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that contains columns to be lagged.
    col_lag_map : dict
        A dictionary where each key is a column name (str),
        and each value is a list of integers specifying the lag offsets.
        Example: { "open": [1, 2, 4], "close": [2] }
    sort_col : str or None, optional
        Sorting strategy before creating lags:
        - None: sort by index.
        - Name of a column: sort by that column.
        - Matches df.index.name: sort by index.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with new lagged feature columns added.
        The new columns are named '<col>_lag_<lag>', preserving original row order.
        Original columns remain in the same order, with new columns appended at the end.
    """

    # Make a copy so we don't mutate the original DataFrame
    df_new = df.copy()

    # Track original row order (index) so we can revert later
    original_index = df_new.index

    # Track original columns so we can keep them in front
    original_columns = list(df_new.columns)

    # Decide how to sort the copied DataFrame
    if sort_col is None:
        # Sort by the DataFrame index
        df_new = df_new.sort_index()
    else:
        # If sort_col is the same as the named index, sort by index
        if sort_col == df_new.index.name:
            df_new = df_new.sort_index()
        else:
            # Otherwise, check if sort_col is an actual column
            if sort_col not in df_new.columns:
                raise ValueError(
                    f"sort_col='{sort_col}' does not match any column or the named index '{df_new.index.name}'."
                )
            df_new = df_new.sort_values(by=sort_col)

    # Create lagged columns
    for col, lag_list in col_lag_map.items():
        if col not in df_new.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
        for lag in lag_list:
            df_new[f"{col}_lag_{lag}"] = df_new[col].shift(lag)

    # Revert to the original row ordering
    df_new = df_new.reindex(original_index)

    # Reorder columns so that the original columns come first
    new_columns = [c for c in df_new.columns if c not in original_columns]
    df_new = df_new[original_columns + new_columns]

    return df_new


def add_time_features(df, date_col):
    """
    Given a DataFrame and a date/time column name, returns a new DataFrame
    that includes various time-based and cyclical features.
    The result preserves the original row order and columns, with
    new columns appended at the end.

    Parameters
    ----------
    df : pandas.DataFrame
        Original dataframe containing the date/time column.
    date_col : str
        Name of the column containing date/time values.

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with additional time-based features,
        leaving the original DataFrame unmodified. The new columns include:
        year, month, day_of_month, day_of_week, day_of_year, quarter,
        hour, minute, day_sin, day_cos, month_sin, month_cos, time_sin, time_cos
    """

    df_new = df.copy()

    # Keep track of original columns
    original_columns = list(df_new.columns)

    # Convert date_col to datetime
    df_new[date_col] = pd.to_datetime(df_new[date_col])

    # Basic time-based features
    df_new["year"] = df_new[date_col].dt.year
    df_new["month"] = df_new[date_col].dt.month
    df_new["day_of_month"] = df_new[date_col].dt.day
    df_new["day_of_week"] = df_new[date_col].dt.weekday  # Monday=0, Sunday=6
    df_new["day_of_year"] = df_new[date_col].dt.dayofyear
    df_new["quarter"] = df_new[date_col].dt.quarter
    df_new["hour"] = df_new[date_col].dt.hour
    df_new["minute"] = df_new[date_col].dt.minute

    # Cyclical encoding for day of week (7 days)
    df_new["day_sin"] = np.sin(2 * np.pi * df_new["day_of_week"] / 7)
    df_new["day_cos"] = np.cos(2 * np.pi * df_new["day_of_week"] / 7)

    # Cyclical encoding for month (12 months)
    df_new["month_sin"] = np.sin(2 * np.pi * df_new["month"] / 12)
    df_new["month_cos"] = np.cos(2 * np.pi * df_new["month"] / 12)

    # Cyclical encoding for hour (24 hours in a day)
    df_new["time_sin"] = np.sin(2 * np.pi * df_new["hour"] / 24)
    df_new["time_cos"] = np.cos(2 * np.pi * df_new["hour"] / 24)

    # Reorder columns so that the original columns come first
    new_columns = [c for c in df_new.columns if c not in original_columns]
    df_new = df_new[original_columns + new_columns]

    return df_new


def add_rolling_statistics_features(
    df,
    col_window_map,
    sort_col=None,
    min_periods=1,
    center=False,
    win_type=None
):
    """
    Creates rolling statistics features for specified columns in a DataFrame,
    ensuring the data is sorted (by index or a given column) before applying
    the rolling window. After creating the rolling features, it reverts the
    DataFrame to the original row ordering.

    For each column and each window size, the following features are created:
      - Rolling mean
      - Rolling sum
      - Rolling minimum
      - Rolling maximum
      - Rolling standard deviation

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that contains columns to be used for rolling stats.
    col_window_map : dict
        A dictionary where each key is a column name (str),
        and each value is a list of window sizes (int) for rolling calculations.
        Example: { "open": [3, 7], "close": [7] }
    sort_col : str or None, optional
        Sorting strategy before creating rolling features:
        - If None, the DataFrame is sorted by index.
        - If it matches the DataFrame's index name, it sorts by index.
        - Otherwise, it sorts by that column.
    min_periods : int, optional (default=1)
        Minimum number of observations in the window required to have a value.
    center : bool, optional (default=False)
        If True, the rolling window is centered. Typically False for time series.
    win_type : str or None, optional
        Type of window (e.g., "triang", "gaussian", "hamming"). If None,
        uses a standard fixed window with no weighting.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with new rolling-statistic feature columns added.
        The new columns follow the naming pattern:
            <col>_roll_<stat>_<window>
        where <stat> ∈ {mean, sum, min, max, std}.
        The final DataFrame has the same row order as the input, with
        original columns in front and new columns appended at the end.
    """

    # Create a copy to avoid mutating the original DataFrame
    df_new = df.copy()

    # Track original row order (index) so we can revert later
    original_index = df_new.index

    # Track original columns to keep them in front
    original_columns = list(df_new.columns)

    # Sorting
    if sort_col is None:
        # Sort by the DataFrame index
        df_new = df_new.sort_index()
    else:
        # If sort_col matches the named index, sort by index
        if sort_col == df_new.index.name:
            df_new = df_new.sort_index()
        else:
            # Otherwise assume it's a column
            if sort_col not in df_new.columns:
                raise ValueError(
                    f"sort_col='{sort_col}' does not match any column "
                    f"or the named index '{df_new.index.name}'."
                )
            df_new = df_new.sort_values(by=sort_col)

    # Create rolling features
    for col, window_list in col_window_map.items():
        if col not in df_new.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

        for window in window_list:
            rolling_obj = df_new[col].rolling(
                window=window,
                min_periods=min_periods,
                center=center,
                win_type=win_type
            )

            df_new[f"{col}_roll_mean_{window}"] = rolling_obj.mean()
            df_new[f"{col}_roll_sum_{window}"]  = rolling_obj.sum()
            df_new[f"{col}_roll_min_{window}"]  = rolling_obj.min()
            df_new[f"{col}_roll_max_{window}"]  = rolling_obj.max()
            df_new[f"{col}_roll_std_{window}"]  = rolling_obj.std()

    # Revert to the original row ordering
    df_new = df_new.reindex(original_index)

    # Reorder columns so that the original columns come first
    new_columns = [c for c in df_new.columns if c not in original_columns]
    df_new = df_new[original_columns + new_columns]

    return df_new

