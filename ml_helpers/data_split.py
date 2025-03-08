import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
):
    """
    Splits a time-series DataFrame using one of:
      1. 'train_test': Single Train-Test split
      2. 'expanding': Multiple splits with growing train set
      3. 'rolling':   Multiple splits with a fixed-size rolling train set

    Sorting Logic (optional):
    -------------------------
    - If sort_col is None, the DataFrame is sorted by its index.
    - If sort_col is the same as the DataFrame's named index, it is also
      sorted by index.
    - Otherwise, it is sorted by the provided column name.

    If 'date_col' is provided, that column is parsed as datetime, but not
    necessarily used for sorting unless 'sort_col == date_col'.

    - Plots each split (Train in blue, Test in orange).
    - Returns a list of (train_df, test_df) pairs.

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
        - (Not used in 'rolling' except for single-split scenario, which we skip.)
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
        - For 'train_test': you'll either get 1 tuple or 0 if invalid.
        - For 'expanding'/'rolling': you may get multiple tuples, or 0 if none valid.
    """
    # 1. Copy the original DataFrame
    df_new = df.copy()

    # Track original index & columns
    original_index = df_new.index
    original_columns = list(df_new.columns)

    # 2. If date_col is provided, parse it as datetime
    if date_col is not None and date_col in df_new.columns:
        df_new[date_col] = pd.to_datetime(df_new[date_col])

    # 3. Sorting logic (similar to your other feature functions)
    if sort_col is None:
        # Sort by index
        df_new = df_new.sort_index()
    else:
        # If sort_col is the named index, sort by index
        if sort_col == df_new.index.name:
            df_new = df_new.sort_index()
        else:
            # Otherwise, check if sort_col is an actual column
            if sort_col not in df_new.columns:
                raise ValueError(
                    f"sort_col='{sort_col}' not found in columns or index name '{df_new.index.name}'."
                )
            df_new = df_new.sort_values(by=sort_col)

    # 4. Determine which columns to plot
    if value_col is None:
        if date_col is not None and date_col in df_new.columns:
            value_cols_to_plot = [c for c in df_new.columns if c != date_col]
        else:
            value_cols_to_plot = list(df_new.columns)
    elif isinstance(value_col, str):
        value_cols_to_plot = [value_col]
    else:
        value_cols_to_plot = list(value_col)

    # For plotting, decide the x-axis
    if date_col is not None and date_col in df_new.columns:
        x_vals = df_new[date_col].values
    else:
        # If no date_col, use the (already reindexed) index
        x_vals = df_new.index.values

    n = len(df_new)

    # A small helper function to plot a portion of the data
    def plot_data(ax, start_idx, end_idx, label_prefix, color):
        """
        Plots the values from start_idx to end_idx for each column in
        value_cols_to_plot, labeling them with label_prefix.
        """
        for col in value_cols_to_plot:
            ax.plot(
                x_vals[start_idx:end_idx],
                df_new[col].iloc[start_idx:end_idx].values,
                label=f"{label_prefix} - {col}",
                color=color
            )

    # We'll store each (train_df, test_df) in a list
    all_splits = []

    # =========================================================================
    # METHOD 1: TRAIN-TEST (Single Split)
    # =========================================================================
    if method == "train_test":
        if 0 < train_size < 1:
            split_idx = int(train_size * n)
        else:
            split_idx = int(train_size)

        # If invalid, return empty list and skip plotting
        if split_idx < 1 or split_idx >= n:
            print("No valid 'train_test' split: 'train_size' out of range.")
            # Revert df_new to original ordering and return
            df_new = df_new.reindex(original_index)
            return []

        train_df = df_new.iloc[:split_idx]
        test_df = df_new.iloc[split_idx:]
        all_splits.append((train_df, test_df))

        # Plot
        plt.figure(figsize=figsize)
        ax = plt.gca()

        # Plot entire dataset in light gray for reference
        for col in value_cols_to_plot:
            ax.plot(x_vals, df_new[col].values, color="lightgray", label=f"All - {col}")

        # Train portion
        plot_data(ax, 0, split_idx, "Train", "blue")
        # Test portion
        plot_data(ax, split_idx, n, "Test", "orange")

        ax.set_title("Train-Test Split")
        ax.set_xlabel(date_col if date_col else "Index")
        ax.set_ylabel("Values")
        ax.legend()
        plt.show()

    # =========================================================================
    # METHOD 2: EXPANDING WINDOW
    # =========================================================================
    elif method == "expanding":
        if 0 < train_size < 1:
            init_train_size = int(train_size * n)
        else:
            init_train_size = int(train_size)

        # Ensure at least 'window_size'
        init_train_size = max(init_train_size, window_size, 1)

        if init_train_size >= n:
            print("No valid 'expanding' splits: 'train_size' >= dataset length.")
            df_new = df_new.reindex(original_index)
            return []

        splits = []
        train_end = init_train_size
        while True:
            test_end = train_end + forecast_horizon
            if test_end > n:
                break
            splits.append((0, train_end, train_end, test_end))
            train_end += step_size

        if not splits:
            print("No valid expanding splits found with the given parameters.")
            df_new = df_new.reindex(original_index)
            return []

        # Plot setup
        rows = int(np.ceil(len(splits) / 2))
        cols = 2 if len(splits) > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*1.5, figsize[1]*rows), squeeze=False)
        axes = axes.flatten()

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            tr_df = df_new.iloc[train_start:train_end]
            te_df = df_new.iloc[test_start:test_end]
            all_splits.append((tr_df, te_df))

            ax = axes[i]
            # Plot entire dataset in light gray
            for col in value_cols_to_plot:
                ax.plot(x_vals, df_new[col].values, color="lightgray")

            # Plot train portion
            plot_data(ax, train_start, train_end, "Train", "blue")
            # Plot test portion
            plot_data(ax, test_start, test_end, "Test", "orange")

            ax.set_title(
                f"Expanding Split #{i}\n"
                f"(Train=[0:{train_end}), Test=[{train_end}:{test_end}))"
            )

        # Hide unused subplots
        for j in range(len(splits), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    # =========================================================================
    # METHOD 3: ROLLING WINDOW
    # =========================================================================
    elif method == "rolling":
        if window_size < 1:
            print("Rolling 'window_size' must be >= 1.")
            df_new = df_new.reindex(original_index)
            return []

        splits = []
        start_idx = 0
        while True:
            train_start = start_idx
            train_end = start_idx + window_size
            test_end = train_end + forecast_horizon
            if test_end > n:
                break
            splits.append((train_start, train_end, train_end, test_end))
            start_idx += step_size

        if not splits:
            print("No valid rolling splits found; possibly window_size + forecast_horizon > length of data.")
            df_new = df_new.reindex(original_index)
            return []

        rows = int(np.ceil(len(splits) / 2))
        cols = 2 if len(splits) > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(figsize[0]*1.5, figsize[1]*rows), squeeze=False)
        axes = axes.flatten()

        for i, (train_start, train_end, test_start, test_end) in enumerate(splits):
            tr_df = df_new.iloc[train_start:train_end]
            te_df = df_new.iloc[test_start:test_end]
            all_splits.append((tr_df, te_df))

            ax = axes[i]
            # Light gray for entire series
            for col in value_cols_to_plot:
                ax.plot(x_vals, df_new[col].values, color="lightgray")

            plot_data(ax, train_start, train_end, "Train", "blue")
            plot_data(ax, test_start, test_end, "Test", "orange")

            ax.set_title(
                f"Rolling Split #{i}\n"
                f"(Train=[{train_start}:{train_end}), Test=[{test_start}:{test_end}))"
            )

        # Hide extra axes
        for j in range(len(splits), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        plt.show()

    else:
        raise ValueError("method must be one of {'train_test', 'expanding', 'rolling'}")

    # 5. Revert df_new to the original row ordering (for consistency)
    df_new = df_new.reindex(original_index)
    # (Note that the returned splits remain in **time-sorted** order.)

    return all_splits

