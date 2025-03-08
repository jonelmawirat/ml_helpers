# ml_helpers

A lightweight Python package providing handy utilities for time-series feature engineering, splitting, and stationarity transformations.

---

## Table of Contents

- [Installation](#installation)
- [Features](#features)
  - [Time-Series Splitting (`data_split.py`)](#time-series-splitting)
  - [Feature Engineering (`feature_engineering.py`)](#feature-engineering)
  - [Stationarity Transformations (`stationarity.py`)](#stationarity-transformations)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
  - [1. Splitting a Time Series](#1-splitting-a-time-series)
  - [2. Adding Lag Features](#2-adding-lag-features)
  - [3. Adding Time-Based Features](#3-adding-time-based-features)
  - [4. Adding Rolling Statistics](#4-adding-rolling-statistics)
  - [5. Making Data Stationary](#5-making-data-stationary)

---

## Installation

```bash
pip install ml_helpers
```

That’s it! You can now import the functions directly from the `ml_helpers` package in your Python scripts or notebooks.

---

## Features

### Time-Series Splitting

- **`split_time_series`**
  Splits your DataFrame in various ways:
  - **Train/Test**: A single split for training and testing.
  - **Expanding**: Multiple splits with a growing training window.
  - **Rolling**: Multiple splits with a fixed-size rolling window.

### Feature Engineering

- **`add_lag_features`**
  Generate lagged features for one or more columns.

- **`add_time_features`**
  Create time-based features like year, month, day_of_week, and cyclical encodings (e.g., sine/cosine for hours, days, months).

- **`add_rolling_statistics_features`**
  Add rolling aggregations (mean, sum, min, max, std) over specified window sizes for your time series columns.

### Stationarity Transformations

- **`make_stationary`**
  Generate log, differenced, and log-differenced versions of one or more columns. Optionally run ADF tests to check stationarity.

---

## Quick Start

1. **Install** the package:
    ```bash
    pip install ml_helpers
    ```

2. **Import** the functions in your script or notebook:
    ```python
    from ml_helpers import (
        split_time_series,
        add_lag_features,
        add_time_features,
        add_rolling_statistics_features,
        make_stationary
    )
    ```

3. **Use** them on your time-series `DataFrame`:
    ```python
    train_test_splits = split_time_series(df, method='train_test', train_size=0.8)
    df_lagged = add_lag_features(df, {'sales': [1, 2]})
    ```

---

## Usage Examples

### 1. Splitting a Time Series

```python
import pandas as pd
from ml_helpers import split_time_series

# Example DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2021-01-01', periods=100, freq='D'),
    'value': range(100)
})

# Single Train-Test split
splits = split_time_series(
    df,
    date_col='date',
    value_col='value',
    method='train_test',
    train_size=0.7
)

# You get a list of (train_df, test_df) pairs. For train_test, it's just 1 pair:
train_df, test_df = splits[0]
```
This plots the train/test split automatically and returns the corresponding subsets.

### 2. Adding Lag Features

```python
import pandas as pd
from ml_helpers import add_lag_features

df = pd.DataFrame({
    'timestamp': pd.date_range(start='2021-01-01', periods=10, freq='D'),
    'value': range(10)
})

# Create lag_1 and lag_2 for 'value'
df_lagged = add_lag_features(
    df,
    col_lag_map={'value': [1, 2]},
    sort_col='timestamp'
)

print(df_lagged.head())
```
Generates new columns: `value_lag_1`, `value_lag_2`.

### 3. Adding Time-Based Features

```python
import pandas as pd
from ml_helpers import add_time_features

df = pd.DataFrame({
    'date': pd.date_range('2021-01-01', periods=5, freq='D'),
    'sales': [10, 15, 20, 25, 30]
})

df_features = add_time_features(df, date_col='date')

print(df_features.columns)
# Original columns plus: year, month, day_of_month, day_of_week, day_of_year,
# quarter, hour, minute, day_sin, day_cos, month_sin, month_cos, time_sin, time_cos
```

### 4. Adding Rolling Statistics

```python
import pandas as pd
from ml_helpers import add_rolling_statistics_features

df = pd.DataFrame({'value': range(10)})
df_rolled = add_rolling_statistics_features(
    df,
    col_window_map={'value': [3]},  # 3-day rolling window
    sort_col=None  # means sort by index
)

print(df_rolled.columns)
# This adds the following columns:
# 'value_roll_mean_3', 'value_roll_sum_3', 'value_roll_min_3',
# 'value_roll_max_3', 'value_roll_std_3'
```

### 5. Making Data Stationary

```python
import pandas as pd
from ml_helpers import make_stationary

# Sample DataFrame
df = pd.DataFrame({
    'date': pd.date_range(start='2021-01-01', periods=5, freq='D'),
    'sales': [100, 110, 115, 120, 150]
})

df_stationary = make_stationary(
    df,
    columns=['sales'],
    sort_col='date',
    run_adf_tests=False  # Change to True to print ADF test results
)

print(df_stationary.columns)
# Adds: sales_log, sales_diff, sales_log_diff
```

