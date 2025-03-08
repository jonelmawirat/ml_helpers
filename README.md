# ML Helpers

A collection of convenient helper functions for simplifying common machine learning workflows and data preprocessing tasks.

## Installation

```bash
pip install ml_helpers
```

## Available Modules and Functions

### Stationarity (`stationarity`)

Functions focused on transforming time series data to stationary series:

- **`make_stationary(df, columns, run_adf_tests=False)`**:
  - Applies log transformations and differencing to convert series into stationary forms.
  - Optionally runs Augmented Dickey-Fuller (ADF) tests to evaluate stationarity.

#### Example Usage

```python
import pandas as pd
from ml_helpers import make_stationary

# Load your time series dataframe
df = pd.read_csv('your_timeseries_data.csv')

# Apply transformations
df_stationary = make_stationary(df, columns=['series1', 'series2'], run_adf_tests=True)
```


### `add_time_features(df, date_col)`

Enhances your DataFrame by adding various basic and cyclical time-based features, such as year, month, day, quarter, hour, and their cyclical representations.

#### Parameters

- **`df`** (`pandas.DataFrame`): Original dataframe containing the date/time column.
- **`date_col`** (`str`): Column name containing date/time values.

#### Returns

- A new DataFrame with additional time-based and cyclical encoded features.

## Example Usage

```python
import pandas as pd
from ml_helpers import add_time_features

# Example DataFrame
df_example = pd.DataFrame({
    'timestamp': pd.date_range('2021-01-01', periods=3, freq='12H')
})

# Generate time features
df_with_features = add_time_features(df_example, 'timestamp')
print(df_with_features)
```
