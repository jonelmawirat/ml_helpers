import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

def make_stationary(df, columns, sort_col=None, run_adf_tests=False):
    """
    For each column in `columns`, this function will create:
      - {col}_log: log of the original series
      - {col}_diff: first difference of the original series
      - {col}_log_diff: first difference of the logged series

    If `run_adf_tests` is True, it will also run ADF tests on each
    of the four series (original, log, diff, log_diff).

    Sorting Logic (optional):
    -------------------------
    - If sort_col is None, the DataFrame is sorted by index before
      computing the transformations.
    - If sort_col is the same as the DataFrame's named index, it also
      sorts by index.
    - Otherwise, it sorts by the specified column.
    - After transformations, the function reverts the DataFrame back to
      the original row ordering.

    Parameters
    ----------
    df : pandas.DataFrame
        The input dataframe containing your time series.
    columns : list
        A list of column names (strings) for which you want to add transformations.
    sort_col : str or None, optional
        A column name to sort by before transformations. If None, sorts by index.
    run_adf_tests : bool, optional
        If True, run the ADF test on each transformation and print the results.

    Returns
    -------
    df_new : pandas.DataFrame
        A new dataframe with new columns added for each transformation,
        preserving original row order and columns (with new columns appended).
    """

    # 1. Make a copy of the original DataFrame
    df_new = df.copy()

    # Track original row order (index) and columns
    original_index = df_new.index
    original_columns = list(df_new.columns)

    # 2. Sorting prior to transformations
    if sort_col is None:
        # Sort by index
        df_new = df_new.sort_index()
    else:
        # If sort_col matches the named index, also sort by index
        if sort_col == df_new.index.name:
            df_new = df_new.sort_index()
        else:
            # Otherwise, assume it's a column
            if sort_col not in df_new.columns:
                raise ValueError(
                    f"sort_col='{sort_col}' does not match any column "
                    f"or the named index '{df_new.index.name}'."
                )
            df_new = df_new.sort_values(by=sort_col)

    # 3. Create transformations for each specified column
    for col in columns:
        if col not in df_new.columns:
            raise ValueError(f"Column '{col}' not found in the DataFrame.")

        # Log of the original series (requires positive values)
        df_new[f"{col}_log"] = np.log(df_new[col])

        # First difference of the original series
        df_new[f"{col}_diff"] = df_new[col].diff()

        # First difference of the logged series
        df_new[f"{col}_log_diff"] = df_new[f"{col}_log"].diff()

    # 4. Revert the DataFrame to the original row ordering
    df_new = df_new.reindex(original_index)

    # 5. Reorder columns so original columns are first, new columns appended
    new_columns = [c for c in df_new.columns if c not in original_columns]
    df_new = df_new[original_columns + new_columns]

    # Optional: Run ADF tests on each transformation
    if run_adf_tests:
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

