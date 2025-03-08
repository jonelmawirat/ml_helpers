import pandas as pd
import numpy as np

def add_time_features(df, date_col):
    """
    Given a DataFrame and a date/time column name, returns a new DataFrame
    that includes various time-based and cyclical features.

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
        leaving the original DataFrame unmodified.
    """
    # Make a copy to avoid mutating the original DataFrame
    df_new = df.copy()

    # Ensure the date_col is in datetime format
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

    return df_new


