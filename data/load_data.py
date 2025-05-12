import os
import pandas as pd
import numpy as np

def load_sp500_data(data_dir='/content/gt_covariance/data/sp500/csv',
                    start_date='2014-01-02',
                    end_date='2019-12-31',
                    features=['return', 'volume', 'high_low_diff']):
    """
    Loads aligned SP500 data from per-stock CSVs where dates are in DD-MM-YYYY format.

    Args:
        data_dir (str): Path to the directory containing CSV files.
        start_date (str): Start date in 'YYYY-MM-DD' format for filtering.
        end_date (str): End date in 'YYYY-MM-DD' format for filtering.
        features (list): List of feature column names to extract.

    Returns:
        data: [timesteps, num_assets, num_features] numpy array
        asset_names: list of stock tickers
    """
    print(f"Loading data from: {data_dir}")
    print(f"Requested date range (YYYY-MM-DD): {start_date} to {end_date}")
    print(f"Expecting date format in CSV files: DD-MM-YYYY")

    try:
        asset_files = sorted(os.listdir(data_dir))
    except FileNotFoundError:
        print(f"Error: Directory not found at {data_dir}")
        raise

    if not asset_files:
        raise ValueError(f"No CSV files found in directory: {data_dir}")

    asset_data = {}
    try:
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
    except ValueError as e:
        print(f"Error parsing date strings: {start_date}, {end_date}. Use YYYY-MM-DD format for function arguments.")
        raise e

    # Generate the target business day index using the datetime objects
    date_index = pd.date_range(start=start_dt, end=end_dt, freq='B')
    print(f"Generated {len(date_index)} target business dates.")

    skipped_assets = []
    loaded_assets_count = 0

    for file in asset_files:
        if not file.lower().endswith('.csv'):
            continue

        path = os.path.join(data_dir, file)
        try:
            df = pd.read_csv(
                path,
                parse_dates=['Date'],
                index_col='Date',
                dayfirst=True  # Tells pandas to interpret DD-MM-YYYY correctly
            )

            # Filter by date *after* reading and parsing
            df = df.sort_index() # Ensure index is sorted for reliable slicing
            # Use the datetime objects for slicing - this is robust
            df = df.loc[start_dt:end_dt]

            if df.empty:
                skipped_assets.append(file)
                continue

            # Use a slightly lower threshold maybe, like 90%? Depends on data quality.
            min_required_days = len(date_index) * 0.90
            if len(df) < min_required_days:
                # print(f"Skipping {file}: Insufficient data coverage ({len(df)}/{len(date_index)} required ~{int(min_required_days)}).")
                skipped_assets.append(file)
                continue

            # Reindex to the common business day index
            df = df.reindex(date_index) # Fill missing business days with NaN

            df['return'] = df['Adjusted Close'].pct_change()
            df['volume'] = df['Volume'] # Keep original volume or fill later
            # Ensure 'Low' is not zero or NaN before dividing
            df['high_low_diff'] = np.where(
                (df['Low'].notna()) & (df['Low'] != 0), # Condition
                (df['High'] - df['Low']) / df['Low'],   # Value if true
                0                                       # Value if false (or NaN)
            )


            # Handle NaNs *after* calculations or reindexing
            df['return'] = df['return'].fillna(0)
            df['volume'] = df['volume'].fillna(method='ffill').fillna(0)
            df['high_low_diff'] = df['high_low_diff'].fillna(method='ffill').fillna(0) # ffill first might be better for diff

            # Ensure only desired features are kept and in correct order
            # Check if all features exist before selecting
            missing_features = [f for f in features if f not in df.columns]
            if missing_features:
                 print(f"Warning: Missing required features {missing_features} in {file} after calculations. Skipping.")
                 skipped_assets.append(file)
                 continue

            asset_data[file.replace('.csv', '')] = df[features]
            loaded_assets_count += 1

        except FileNotFoundError:
            print(f"Warning: File not found {path} - skipping.")
            skipped_assets.append(file)
        except KeyError as e:
            # This might catch issues if required columns like 'Adjusted Close', 'Volume' etc. are missing
            print(f"Warning: KeyError '{e}' processing {file}. Missing expected column in CSV? Skipping.")
            skipped_assets.append(file)
        except ValueError as e:
             # This might catch date parsing errors if a date is malformed even with dayfirst=True
             print(f"Warning: ValueError processing {file} (likely date format issue): {e}. Skipping.")
             skipped_assets.append(file)
        except Exception as e:
            print(f"Warning: Unexpected error processing {file}: {type(e).__name__} - {e}. Skipping.")
            skipped_assets.append(file)


    asset_names = sorted(asset_data.keys())
    print(f"Successfully loaded data for {len(asset_names)} assets.")
    if skipped_assets:
        print(f"Skipped {len(skipped_assets)} assets (due to missing data, errors, or coverage).") # First few: {skipped_assets[:5]}")

    if not asset_names:
        raise ValueError("No usable assets found after filtering. Check source data, date range, format (DD-MM-YYYY expected), required columns, and coverage threshold.")

    # Build final array (same as before)
    num_timesteps = len(date_index)
    num_assets = len(asset_names)
    num_features = len(features)

    data = np.zeros((num_timesteps, num_assets, num_features), dtype=np.float32)
    for i, ticker in enumerate(asset_names):
        df_ticker = asset_data[ticker]
        if df_ticker.shape == (num_timesteps, num_features):
            # Check for NaNs / Infs before assigning
            if df_ticker.isnull().values.any() or np.isinf(df_ticker.values).any():
                print(f"Warning: NaN or Inf values found in final data for {ticker}. Consider refining NaN handling. Filling with 0 for now.")
                data[:, i, :] = df_ticker.fillna(0).replace([np.inf, -np.inf], 0).values
            else:
                data[:, i, :] = df_ticker.values
        else:
             print(f"Error: Shape mismatch for {ticker}. Expected {(num_timesteps, num_features)}, Got {df_ticker.shape}. Filling with zeros.")

    if np.isnan(data).any() or np.isinf(data).any():
        print("Warning: Final data array contains NaN or Inf values!")

    return data, asset_names