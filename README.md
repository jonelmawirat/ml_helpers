1. **Installation**
2. **Overview**
3. **Usage Examples**
    - **data_split.py**
        - `split_time_series()`
            - Train-Test example
            - Expanding example
            - Rolling example
    - **feature_engineering.py**
        - `add_lag_features()`
        - `add_time_features()`
        - `add_rolling_statistics_features()`
    - [**stationarity.py**](http://stationarity.py/)
        - `make_stationary()`

---

## 1. Installation

If you haven’t already, you can install the library from PyPI using:

```bash
pip install jmawirat-ml-helpers

```

After installation, you can import its functions directly in your Python scripts or notebooks.

---

## 2. Overview

The **jmawirat-ml-helpers** package provides utility functions to:

- **Split time-series data** into train/test sets (single split), multiple expanding windows, or rolling windows.
- **Engineer features** such as lag features, rolling statistics, and time-based features (e.g., cyclical encodings for day/week/month).
- **Assess and enhance stationarity** by creating log-transformed and differenced versions of a time-series, with optional Augmented Dickey-Fuller (ADF) tests.

These helpers streamline the time-series modeling process from data preparation to stationarity checks.

---

## 3. Usage Examples

Below, we showcase each module, its main functions, parameters, and typical usage patterns.

---

### A. `data_split.py`

### `split_time_series()`

```python
def split_time_series(
    df,
    date_col=None,
    value_col=None,
    method="train_test",
    train_size=0.7,
    window_size=50,
    forecast_horizon=10,
    step_size=1,
    figsize=(10, 6),
    sort_col=None
) -> list:
    """
    Splits a time-series DataFrame using one of:
      1. 'train_test': Single Train-Test split
      2. 'expanding': Multiple splits with growing train set
      3. 'rolling':   Multiple splits with a fixed-size rolling train set

    Parameters
    ----------
    df : pandas.DataFrame
        Your dataset. If it's time-series, ensure it's properly indexed or
        pass 'sort_col' to sort by a particular column. If 'date_col' is given,
        that column will be parsed as datetime (but not necessarily sorted on
        unless 'sort_col' matches it).
    date_col : str, optional
        Name of the column containing dates (for plotting). If None, the function
        will use the df's index for the x-axis in plots.
    value_col : str or list, optional
        Column(s) to plot. If None, plots ALL columns (except 'date_col' if provided).
        If a single string is provided, it's converted to a list internally.
    method : str, optional
        One of {'train_test', 'expanding', 'rolling'}.
    train_size : float or int, optional
        - For 'train_test': fraction (0 < train_size < 1) or row count of the training set.
        - For 'expanding': fraction or row count for the initial training set.
        - (Not used in 'rolling' except for single-split scenario, which is typically not done.)
    window_size : int, optional
        The size of the training window (for rolling), or minimal train size (expanding).
    forecast_horizon : int, optional
        Number of rows to include in each test set.
    step_size : int, optional
        How many rows to move for each new split in 'expanding'/'rolling' methods.
    figsize : tuple, optional
        Size of the figure for each visualization.
    sort_col : str or None, optional
        If None, sort by index. Otherwise, sort by 'sort_col' unless
        it matches the named index, in which case it's sorted by index.

    Returns
    -------
    list of (train_df, test_df)
        Each element is a tuple (train_df, test_df) for one split.
        - For 'train_test': you'll get 1 tuple (or 0 if invalid).
        - For 'expanding'/'rolling': you'll get multiple tuples (or 0 if invalid).
    """
    ...

```

**Key Points**

- Automatically plots each split for easy visualization.
- Returns a list of `(train_df, test_df)` pairs.
- The `sort_col` parameter ensures your data is sorted correctly before splitting.
- After splitting, the function reverts the DataFrame to its **original row order**, so subsequent operations on `df` remain consistent.

### Examples

Below are example usages for each `method`:

---

### 1. **Train-Test Split**

```python
import pandas as pd
from jmawirat_ml_helpers.data_split import split_time_series

# Assume you have a time-series DataFrame df
# with an optional date column called 'Date' and a value column called 'Value'
splits = split_time_series(
    df=df,
    date_col='Date',       # for x-axis plotting
    value_col='Value',     # could be a list or None
    method='train_test',   # <--- specifying train_test split
    train_size=0.8,        # 80% of rows as training set
    sort_col='Date'        # ensures data is sorted by 'Date' before splitting
)

# splits will be a list with one tuple: [(train_df, test_df)]
train_df, test_df = splits[0]

print("Train shape:", train_df.shape)
print("Test shape:",  test_df.shape)

```

- **train_size** can be either a fraction (0 < train_size < 1) or an integer specifying the number of training rows.

---

### 2. **Expanding Window Split**

```python
from jmawirat_ml_helpers.data_split import split_time_series

splits = split_time_series(
    df=df,
    date_col='Date',
    value_col='Value',
    method='expanding',    # <--- specifying expanding window splits
    train_size=0.2,        # 20% of the data as the initial training window
    window_size=50,        # minimum training size if 0.2 * n < 50
    forecast_horizon=10,   # each test set will have 10 rows
    step_size=5,           # the training set expands, but we step 5 rows at a time
    sort_col='Date'
)

# splits is now a list of (train_df, test_df) pairs
for i, (train_df, test_df) in enumerate(splits):
    print(f"Split #{i}: Train={train_df.shape}, Test={test_df.shape}")

```

- The `train_size` controls the **initial** training window. Subsequent windows expand by `step_size` rows at a time.
- Each test set is `forecast_horizon` rows.

---

### 3. **Rolling Window Split**

```python
from jmawirat_ml_helpers.data_split import split_time_series

splits = split_time_series(
    df=df,
    date_col='Date',
    value_col='Value',
    method='rolling',     # <--- rolling window
    window_size=100,      # fixed training size
    forecast_horizon=10,
    step_size=10,         # roll forward 10 rows at a time
    sort_col='Date'
)

for i, (train_df, test_df) in enumerate(splits):
    print(f"Rolling Split #{i}:")
    print("  Train indices:", train_df.index.min(), "->", train_df.index.max())
    print("  Test indices :", test_df.index.min(),  "->", test_df.index.max())

```

- The **training window** is always of length `window_size`.
- Each test set is `forecast_horizon` rows.
- The window slides forward by `step_size` for each new split.

---

### B. `feature_engineering.py`

This module provides functions to create new features based on existing columns—useful in many time-series and classical ML workflows.

### 1. `add_lag_features()`

```python
def add_lag_features(df, col_lag_map, sort_col=None):
    """
    Creates lagged features for specified columns in a DataFrame.
    - Sorts the data (by index or a given column) before creating lags.
    - Reverts to original order after feature creation.
    - New columns are named '<col>_lag_<lag>'.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame that contains columns to be lagged.
    col_lag_map : dict
        A dictionary where each key is a column name, and each value is a
        list of integers specifying the lag offsets.
        Example: { "open": [1, 2, 4], "close": [2] }
    sort_col : str or None, optional
        Sorting strategy before creating lags.
    Returns
    -------
    pandas.DataFrame
        A DataFrame with new lagged feature columns added, original columns first.
    """
    ...

```

**Example**

```python
from jmawirat_ml_helpers.feature_engineering import add_lag_features

data = {
    'Date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'Value': [10, 12, 15, 14, 18]
}
df = pd.DataFrame(data)

col_lag_map = {'Value': [1, 2]}  # create Value_lag_1 and Value_lag_2
df_with_lags = add_lag_features(df, col_lag_map, sort_col='Date')

print(df_with_lags)

```

Output (sketch):

```
        Date  Value  Value_lag_1  Value_lag_2
0 2021-01-01     10          NaN          NaN
1 2021-01-02     12         10.0          NaN
2 2021-01-03     15         12.0         10.0
3 2021-01-04     14         15.0         12.0
4 2021-01-05     18         14.0         15.0

```

---

### 2. `add_time_features()`

```python
def add_time_features(df, date_col):
    """
    Given a DataFrame and a date/time column, returns a new DataFrame
    with various time-based and cyclical features:
        - year, month, day_of_month, day_of_week, day_of_year, quarter,
        - hour, minute,
        - day_sin, day_cos (cyclical for day_of_week),
        - month_sin, month_cos (cyclical for month),
        - time_sin, time_cos (cyclical for hour).

    Parameters
    ----------
    df : pandas.DataFrame
    date_col : str

    Returns
    -------
    pandas.DataFrame
        A copy of the original DataFrame with new columns appended.
    """
    ...

```

**Example**

```python
from jmawirat_ml_helpers.feature_engineering import add_time_features

data = {
    'Date': pd.date_range(start='2021-01-01', periods=3, freq='H'),
    'Value': [10, 12, 15]
}
df = pd.DataFrame(data)

df_time = add_time_features(df, date_col='Date')
print(df_time.columns)

```

Output columns (sketch):

```
Index([
  'Date',
  'Value',
  'year', 'month', 'day_of_month', 'day_of_week',
  'day_of_year', 'quarter', 'hour', 'minute',
  'day_sin', 'day_cos', 'month_sin', 'month_cos',
  'time_sin', 'time_cos'
], dtype='object')

```

---

### 3. `add_rolling_statistics_features()`

```python
def add_rolling_statistics_features(
    df,
    col_window_map,
    sort_col=None,
    min_periods=1,
    center=False,
    win_type=None
):
    """
    Creates rolling statistics features for specified columns in a DataFrame:
      - Rolling mean, sum, min, max, std

    Parameters
    ----------
    df : pandas.DataFrame
    col_window_map : dict
        { "open": [3, 7], "close": [7] } means for 'open', it creates
        3-day and 7-day rolling stats, for 'close', it creates 7-day rolling stats.
    sort_col : str or None
    min_periods : int, default 1
    center : bool, default False
    win_type : str or None

    Returns
    -------
    pandas.DataFrame
        A DataFrame with new rolling-stat features appended.
    """
    ...

```

**Example**

```python
from jmawirat_ml_helpers.feature_engineering import add_rolling_statistics_features

data = {
    'Date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'Value': [10, 12, 15, 14, 18]
}
df = pd.DataFrame(data)

col_window_map = {'Value': [2, 3]}  # 2-day and 3-day rolling
df_rolled = add_rolling_statistics_features(
    df,
    col_window_map,
    sort_col='Date',
    min_periods=1
)

print(df_rolled.columns)

```

You’ll see columns like:

```
[
 'Date', 'Value',
 'Value_roll_mean_2', 'Value_roll_sum_2', 'Value_roll_min_2',
 'Value_roll_max_2', 'Value_roll_std_2',
 'Value_roll_mean_3', 'Value_roll_sum_3', 'Value_roll_min_3',
 'Value_roll_max_3', 'Value_roll_std_3'
]

```

---

### C. `stationarity.py`

### `make_stationary()`

```python
def make_stationary(df, columns, sort_col=None, run_adf_tests=False):
    """
    For each column in `columns`, this function will create:
      - {col}_log: log of the original series
      - {col}_diff: first difference of the original series
      - {col}_log_diff: first difference of the logged series

    If `run_adf_tests` is True, it runs the Augmented Dickey-Fuller test
    on each of the four series (original, log, diff, log_diff).

    Parameters
    ----------
    df : pandas.DataFrame
    columns : list of str
    sort_col : str or None
    run_adf_tests : bool

    Returns
    -------
    pandas.DataFrame
        A new dataframe with the new columns appended.
    """
    ...

```

**Example**

```python
from jmawirat_ml_helpers.stationarity import make_stationary

data = {
    'Date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'Value': [10, 12, 15, 14, 18]
}
df = pd.DataFrame(data)

df_stationary = make_stationary(
    df,
    columns=['Value'],
    sort_col='Date',
    run_adf_tests=True  # prints out ADF statistics
)

print(df_stationary.columns)

```

You should see:

```
[
 'Date', 'Value',
 'Value_log',
 'Value_diff',
 'Value_log_diff'
]

```

And if `run_adf_tests=True`, it will print something like:

```
========== ADF Tests for 'Value' ==========
=== Original Series ===
ADF Statistic: <some_value>
p-value: <some_value>
Critical Values: {...}

=== Log Series ===
ADF Statistic: <some_value>
p-value: <some_value>
Critical Values: {...}

=== Differenced Series ===
...
------------------------------------

```

This helps you identify which transformations are stationary.

---

## Putting It All Together

A typical time-series workflow using **jmawirat-ml-helpers** might look like:

1. **Load and inspect** your DataFrame.
2. **Split** the DataFrame into train/test sets (or multiple folds via expanding/rolling).
3. **Feature engineering** on the training set (e.g., lag features, rolling stats, time-based features).
4. **Check stationarity** (create differenced or log-differenced columns, run ADF tests).
5. **Train your model** on these engineered features.
6. **Evaluate** on the test set.

Example snippet:

```python
import pandas as pd
from jmawirat_ml_helpers.data_split import split_time_series
from jmawirat_ml_helpers.feature_engineering import (
    add_lag_features,
    add_time_features,
    add_rolling_statistics_features
)
from jmawirat_ml_helpers.stationarity import make_stationary

df = pd.read_csv('my_time_series.csv')

# 1. Split data
train_df, test_df = split_time_series(df, date_col='Date', value_col='Value',
                                      method='train_test', train_size=0.8,
                                      sort_col='Date')[0]

# 2. Add time features
train_df = add_time_features(train_df, 'Date')
test_df  = add_time_features(test_df, 'Date')

# 3. Add lag features
train_df = add_lag_features(train_df, {'Value': [1, 2]}, sort_col='Date')
test_df  = add_lag_features(test_df, {'Value': [1, 2]}, sort_col='Date')

# 4. Rolling stats
train_df = add_rolling_statistics_features(train_df, {'Value': [3]}, sort_col='Date')
test_df  = add_rolling_statistics_features(test_df, {'Value': [3]}, sort_col='Date')

# 5. Check stationarity (optional)
train_df = make_stationary(train_df, ['Value'], sort_col='Date', run_adf_tests=True)

# Now train your model using train_df, then evaluate on test_df
...

```

