import numpy as np
from statsmodels.tsa.stattools import adfuller

def make_stationary(df, columns, run_adf_tests=False):
    """
    For each column in `columns`, this function will create:
      - {col}_log: log of the original series
      - {col}_diff: first difference of the original series
      - {col}_log_diff: first difference of the logged series

    If `run_adf_tests` is True, it will also run ADF tests on each
    of the four series (original, log, diff, log_diff).

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing your time series.
    columns : list
        A list of column names (strings) for which you want to add transformations.
    run_adf_tests : bool, optional
        If True, run the ADF test on each transformation and print the results.

    Returns
    -------
    df_new : pandas.DataFrame
        A new dataframe with new columns added for each transformation.
    """
    # Make a copy of the original dataframe so it won't be mutated
    df_new = df.copy()

    for col in columns:
        # 1. Log transform
        df_new[f"{col}_log"] = np.log(df_new[col])

        # 2. First difference of original series
        df_new[f"{col}_diff"] = df_new[col].diff()

        # 3. First difference of the log series
        df_new[f"{col}_log_diff"] = df_new[f"{col}_log"].diff()

    if run_adf_tests:
        # Perform ADF tests on each transformation for each column
        for col in columns:
            print(f"========== ADF Tests for '{col}' ==========")

            # Original series
            adf_original = adfuller(df_new[col].dropna())
            print("=== Original Series ===")
            print("ADF Statistic:", adf_original[0])
            print("p-value:", adf_original[1])
            print("Critical Values:", adf_original[4])
            print()

            # Log series
            adf_log = adfuller(df_new[f"{col}_log"].dropna())
            print("=== Log Series ===")
            print("ADF Statistic:", adf_log[0])
            print("p-value:", adf_log[1])
            print("Critical Values:", adf_log[4])
            print()

            # Differenced series
            adf_diff = adfuller(df_new[f"{col}_diff"].dropna())
            print("=== Differenced Series ===")
            print("ADF Statistic:", adf_diff[0])
            print("p-value:", adf_diff[1])
            print("Critical Values:", adf_diff[4])
            print()

            # Log-differenced series
            adf_log_diff = adfuller(df_new[f"{col}_log_diff"].dropna())
            print("=== Log-Differenced Series ===")
            print("ADF Statistic:", adf_log_diff[0])
            print("p-value:", adf_log_diff[1])
            print("Critical Values:", adf_log_diff[4])
            print("------------------------------------\n")

    return df_new

