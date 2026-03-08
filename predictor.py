import fastf1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn as skl
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import joblib

CSV_PATH = '/Users/ajayshah/FastF1 Project/historical_data.csv'
CACHE_PATH = '/Users/ajayshah/FastF1 Project/'
MODEL_PATH = '/Users/ajayshah/FastF1 Project/model.pkl'

# Flag to retrain the model
RETRAIN_MODEL = True

# Create the folder if it doesn't exist
if not os.path.exists(CACHE_PATH):
    os.makedirs(CACHE_PATH)

# Enable cache
fastf1.Cache.enable_cache(CACHE_PATH)


def get_past_race(year:int, max_round=None)->pd.DataFrame:
    '''
    Fetches the the past race and qualifying for the specified year, up to the max round

    Parameters:
    -----------
    int: year
        The year to get data for
    int: max_round (optional parameter)
        The maximum round number to gather data for
    
    Returns:
    --------
    pd.DataFrame: races_df
        A DataFrame containing all of the past race and qualifying data
    '''
    # Gets the schedule for the specified year, excluding testing
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    # Gets the number of rounds in the specified year
    num_rounds = len(schedule)

    # Get up to a cetain round
    if max_round:
        num_rounds = min(max_round, num_rounds)

    # Load existing CSV if it exists, otherwise start fresh
    if os.path.exists(CSV_PATH):
        existing_df = pd.read_csv(CSV_PATH)
        existing_rounds = set(zip(existing_df['Year'], existing_df['Round']))
        race_data = [existing_df]
        print("Existing data loaded from CSV")
    else:
        existing_rounds = set()
        race_data = []
        print("No CSV found, fetching all data from scratch")

    # Iterates over each round to check and see if the data is already saved into the CSV
    for i in range(1, num_rounds + 1):
        if (year, i) in existing_rounds:
            print(f"Round {i} already in CSV, skipping...")
            continue
        try:
            print(f"Fetching round {i} of {num_rounds}...")

            # Load race session
            curr_race = fastf1.get_session(year, i, 'R')
            curr_race.load()
            results = curr_race.results.copy()
            results['Year'] = year
            results['Round'] = i

            # Load qualifying session
            curr_quali = fastf1.get_session(year, i, 'Q')
            curr_quali.load()
            quali_results = curr_quali.results.copy()

            # Use row order to derive grid position
            quali_results = quali_results.reset_index(drop=True)
            quali_results['DerivedGridPosition'] = quali_results.index + 1

            # Only keep what we need from qualifying
            quali_results = quali_results[['DriverId', 'DerivedGridPosition']]

            # Merge into race results
            results = results.merge(quali_results, on='DriverId', how='left')

            race_data.append(results)

        except Exception as e:
            print(f"Skipping round {i}: {e}")

    # Combine everything and save
    if race_data:
        races_df = pd.concat(race_data, ignore_index=True)
        races_df.to_csv(CSV_PATH, index=False)
        print("Data saved to CSV")
    else:
        races_df = pd.DataFrame()
        print("No data to save")

    return races_df


def engineer_features(df:pd.DataFrame)->pd.DataFrame:
    '''
    Transforms the raw race data by adding rolling and cumulative features

    Parameters:
    -----------
    pd.DataFrame: df
        The raw race DataFrame to engineer features for
    
    Returns:
    --------
    pd.DataFrame: df
        The DataFrame with new columns added
    '''
    # Sort the DataFrame by 'Year' and 'Round' 
    df = df.sort_values(by=['Year', 'Round']).reset_index(drop=True)

    # Calulate the rolling average of driver's last 3 finishing positions
    df['AvgFinishes'] = df.groupby('DriverId', sort=False)['Position'].shift(1).rolling(window=3, min_periods=1).mean()

    # Shift points by 1
    df['PointsShifted'] = df.groupby(['DriverId', 'Year'], sort=False)['Points'].shift(1)
    # Calculate the cumulative total
    df['CumulativePoints'] = df.groupby(['DriverId', 'Year'], sort=False)['PointsShifted'].cumsum()

    # Calculate the average finishing position for both drivers of a team per race
    df['TeamAvgFinish'] = df.groupby(['TeamName', 'Round', 'Year'])['Position'].transform('mean')

    # Shift team average by 1
    df['TeamAvgFinishShifted'] = df.groupby(['TeamName', 'Year'], sort=False)['TeamAvgFinish'].shift(1)
    # Calculate the rolling average over the last 3 races
    df['TeamRollingAvg'] = df.groupby(['TeamName', 'Year'], sort=False)['TeamAvgFinishShifted'].rolling(window=3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

    # Fill NaNs with 0 for points 
    df['CumulativePoints'] = df['CumulativePoints'].fillna(0)
    # Fill NaNs with 10.5 (lower half finishing position)
    df['TeamAvgFinish'] = df['TeamAvgFinish'].fillna(10)
    df = df.fillna(10.5)

    # Drop unneeded column
    df = df.drop(columns=['TeamAvgFinishShifted'])

    # Return the DataFrame
    return df


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    '''
    Cleans the inputted DataFrame

    Parameters:
    -----------
    pd.DataFrame: df
        The DataFrame to clean
    
    Returns:
    --------
    pd.DataFrame: df
        The cleaned DataFrame
    '''
    # Drop unneeded columns
    cols_to_drop = ['BroadcastName', 'FirstName', 'LastName', 'FullName',
                    'HeadshotUrl', 'TeamColor', 'TeamId', 'CountryCode',
                    'Time', 'Abbreviation', 'Q1', 'Q2', 'Q3', 'PointsShifted', 'DriverNumber', 'TeamAvgFinish', 'Points', 'Finished']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Simplify status to binary
    df['Finished'] = (df['Status'] == 'Finished').astype(int)
    df = df.drop(columns=['Status'])

    # Convert ClassifiedPosition to numeric
    df['ClassifiedPosition'] = pd.to_numeric(df['ClassifiedPosition'], errors='coerce')

    # Fill all NaN data with 20 (indicating the driver(s) finishing last)
    df['ClassifiedPosition'] = df['ClassifiedPosition'].fillna(20)

    # Standardize team names so that even if the name of the team changes, the car is still viewed as the same
    df['TeamName'] = df['TeamName'].replace({
        'RB': 'Racing Bulls',
        'Kick Sauber': 'Audi'
    })

    # One-hot encode the TeamName and DriverId columns
    df = pd.get_dummies(df, columns=['TeamName', 'DriverId'], dtype=int)

    # Returns the DataFrame
    return df


def predict_race(year:int, round:int, raw_df:pd.DataFrame, x_train:pd.DataFrame)->tuple[pd.DataFrame, np.ndarray]:
    '''
    Predicts the results of the race of the entered round

    Parameters:
    -----------
    int: year
        The season of the race to predict
    int: round
        The round to predict
    pd.DataFrame: raw_df
        The DataFrame containing all the data before it gets cleaned 
    pd.DataFrame: x_train
        The training feature matrix
    
    Returns:
    --------
    pd.DataFrame: prediction_features
        The features needed for prediction
    np.ndarray: driver_ids
        The Ids of the drivers 
    '''
    # Load qualifying session
    quali = fastf1.get_session(year, round, 'Q')
    quali.load()
    # Save the qualifying data to the prediction_features DataFrame
    prediction_features = quali.results.copy()

    # Use row order to derive grid position
    prediction_features = prediction_features.reset_index(drop=True)
    prediction_features['DerivedGridPosition'] = prediction_features.index + 1

    # Filter the raw_df to include only the races before the specified round
    filtered_df = raw_df[(raw_df['Year'] < year) | ((raw_df['Year'] == year) & (raw_df['Round'] < round))]

    # Sort by 'Year' and 'Round' to chronologically order the data
    filtered_df = filtered_df.sort_values(by=['Year', 'Round']).reset_index(drop=True)
    filtered_df = filtered_df.groupby(['DriverId']).last().reset_index()

    # Keep only the data needed for prediction
    filtered_df = filtered_df[['DriverId', 'AvgFinishes', 'CumulativePoints', 'TeamRollingAvg']]

    # Merge the historical features onto the qualifying data
    prediction_features = prediction_features.merge(filtered_df, on='DriverId', how='left')

    # Save the driver Ids before the encoding removes them
    driver_ids = prediction_features['DriverId'].values

    # One-hot encode TeamName and DriverId to match training data format
    prediction_features = pd.get_dummies(prediction_features, columns=['TeamName', 'DriverId'], dtype=int)

    # Align columns to exactly match x_train, filling missing columns with 0
    prediction_features = prediction_features.reindex(columns=x_train.columns, fill_value=0)
    print(driver_ids)
    # Return the prediction_features and the driver_ids
    return prediction_features, driver_ids


def main():
    # Fetch the data from previous season/races
    races_df = get_past_race(2025)

    # Call the engineer_features function
    races_df = engineer_features(races_df)
    # Copy the DataFrame to the raw DataFrame
    raw_df = races_df.copy()

    # Clean the data
    races_df = clean_data(races_df)

    # Define the features and the target
    y = races_df['Position']
    x = races_df.drop(columns=['Position', 'ClassifiedPosition', 'Finished'])

    # Train/test split
    # Train on 2025 and the first round of 2026 data
    x_train = x[(x['Year'] == 2025) | ((x['Year'] == 2026) & (x['Round'] < 2))]
    x_test = x[(x['Year'] == 2026) & (x['Round'] >= 2)]
    y_train = y.loc[x_train.index]
    y_test  = y.loc[x_test.index]

    x_train = x_train.drop(columns=['Year'])
    x_test  = x_test.drop(columns=['Year'])

    # Train or load model
    if RETRAIN_MODEL and os.path.exists(MODEL_PATH):
        os.remove(MODEL_PATH)

    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        print("Model loaded from file")
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(x_train, y_train)
        joblib.dump(model, MODEL_PATH)
        print("Model trained and saved")

    # Evaluate the model on the test set
    '''
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_test, y_pred)
    print(f"Test set MAE: {mae:.2f}")

    importances = pd.Series(model.feature_importances_, index=x_train.columns)
    print(importances.sort_values(ascending=False))
    '''

    # Predict the race
    prediction_features, driver_ids = predict_race(2026, 1, raw_df, x_train)
    predictions = model.predict(prediction_features)

    results = pd.DataFrame({
        'Driver': driver_ids,
        'PredictedFinish': predictions
    })
    results = results.sort_values('PredictedFinish').reset_index(drop=True)
    results.index += 1
    print(results)


if __name__ == '__main__':
    main()
