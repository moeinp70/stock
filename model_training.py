import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import timedelta
import pandas as pd
import matplotlib.pyplot as plt
import random
import tensorflow as tf
# Step 1: Create Supervised Dataset (LSTM: reshape into 3D - samples, timesteps, features)
# Set the seed for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)

set_random_seeds(42)
# Step 1: Create Supervised Dataset (LSTM: reshape into 3D - samples, timesteps, features)
def create_supervised_dataset(merged_data, specific_date, window_size, prediction_horizon, scaler=None):
    X = []
    y = []
    dates = []
    set_random_seeds(42)

    # Filter the data to only use data before the specific date for training
    filtered_data = merged_data[merged_data['date'] < specific_date]

    # Define the features list based on your columns
    features = ['y', 'Total Revenue', 'Net Income', 'Stockholders Equity', 'Total Liabilities', 'Total Assets']

    # Normalize the features using the provided scaler
    scaled_data = filtered_data.copy()
    if scaler:
        scaled_data[features] = scaler.fit_transform(filtered_data[features])

    for i in range(window_size, len(scaled_data) -prediction_horizon+1 ):
        # Input: last 'window_size' days of stock prices and financial data
        X_window = scaled_data.iloc[i-window_size:i][features].values
        X.append(X_window)

        # Output: next 'prediction_horizon' days of stock prices
        y_window = scaled_data.iloc[i:i+prediction_horizon]['y'].values
        y.append(y_window)

        # Collect dates corresponding to the labels (last date in the prediction window)
        dates.append(scaled_data.iloc[i-1]['date'])
    for j in range(prediction_horizon-1):
        dates.append(scaled_data.iloc[i+j]['date'])
    dates_all = merged_data['date']
    X = np.array(X)  # Shape: (n_samples, window_size, n_features)
    y = np.array(y)  # Shape: (n_samples, prediction_horizon)

    return X, y, dates, dates_all, merged_data['y'].values

# Step 2: Train the LSTM Model (time series prediction)
# Updated LSTM training function to support fine-tuning


def create_supervised_dataset1(merged_data, specific_date, window_size, prediction_horizon, scaler=None):
    X = []
    y = []
    dates = []
    set_random_seeds(42)

    # Filter the data to only use data before the specific date for training
    filtered_data = merged_data[merged_data['date'] < specific_date]

    # Define the features list based on your columns
    features = ['y', 'Total Revenue', 'Net Income', 'Stockholders Equity', 'Total Liabilities', 'Total Assets']

    # Normalize the features using the provided scaler
    scaled_data = filtered_data.copy()
    if scaler:
        scaled_data[features] = scaler.fit_transform(filtered_data[features])

    # Start the loop from the end of the dataset and go backward to ensure the latest data is included
    for i in range(len(scaled_data) - prediction_horizon, window_size+prediction_horizon , -1):
        # Input: last 'window_size' days of stock prices and financial data
        X_window = scaled_data.iloc[i - window_size:i][features].values
        X.append(X_window)

        # Output: next 'prediction_horizon' days of stock prices
        y_window = scaled_data.iloc[i:i + prediction_horizon]['y'].values
        y.append(y_window)

        # Collect dates corresponding to the labels (last date in the prediction window)
        dates.append(scaled_data.iloc[i + prediction_horizon-1]['date'])

    dates_all = merged_data['date']
    X = np.array(X)  # Shape: (n_samples, window_size, n_features)
    y = np.array(y)  # Shape: (n_samples, prediction_horizon)

    return X, y, dates, dates_all, merged_data['y'].values




def train_lstm_model(X_train, y_train, window_size, prediction_horizon, model=None, epochs=50, batch_size=32):
    """
    If model is provided, fine-tune it.
    If no model is provided, create a new LSTM model and train it.
    """
    n_features = X_train.shape[2]  # The number of features in the dataset
    set_random_seeds(42)

    if model is None:
        # If no model is passed, create a new model
        model = Sequential()
        model.add(LSTM(units=128, return_sequences=False, input_shape=(window_size, n_features)))
        #model.add(LSTM(units=64, return_sequences=True, input_shape=(window_size, n_features)))

        # Second LSTM layer without `return_sequences`
        #model.add(LSTM(units=64, return_sequences=False))
        #model.add(Dropout(0.2))

        model.add(Dense(units=prediction_horizon))  # Predict 'prediction_horizon' days ahead
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("Training new model from scratch.")

    else:
        print("Fine-tuning the existing model.")

    # Continue training or fine-tuning the model
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    return model

# Step 3: Predict future stock prices after the specific date
def predict_from_date(trained_model, merged_data, specific_date, prediction_horizon, window_size, scaler=None):
    features = ['y', 'Total Revenue', 'Net Income', 'Stockholders Equity', 'Total Liabilities', 'Total Assets']

    specific_date = pd.to_datetime(specific_date)

    train_data = merged_data[merged_data['date'] < specific_date].copy()

    # Use the last 'window_size' days for generating test features
    last_data = train_data.tail(window_size).copy()

    if scaler:
        last_data[features] = scaler.transform(last_data[features])

    # Prepare features for prediction (reshape to 3D)
    X_test = last_data[features].values.reshape(1, window_size, len(features))

    # Predict the stock prices for the next 'prediction_horizon' days
    y_pred_scaled = trained_model.predict(X_test).flatten()

    # Inverse transform the predicted values if scaler is provided
    if scaler:
        y_pred_full = np.zeros((y_pred_scaled.shape[0], len(features)))
        y_pred_full[:, 0] = y_pred_scaled  # Only the 'y' column needs to be inverse transformed
        y_pred_inversed = scaler.inverse_transform(y_pred_full)[:, 0]
    else:
        y_pred_inversed = y_pred_scaled

    future_dates = pd.date_range(start=specific_date + timedelta(days=1), periods=prediction_horizon, freq='B')

    return future_dates, y_pred_inversed

# Step 4: Plot the Results
def plot_results(validation_dates, dates_all, y_val_pred, y_val_true, prediction_dates=None, y_pred=None, merged_data=None, specific_date=None, scaler=None):
    plt.figure(figsize=(14, 7))

    features = ['y', 'Total Revenue', 'Net Income', 'Stockholders Equity', 'Total Liabilities', 'Total Assets']

    y_val_pred_full = np.zeros((len(y_val_pred), len(features)))
    y_val_pred_full[:, 0] = y_val_pred

    if scaler:
        y_val_pred_rescaled = scaler.inverse_transform(y_val_pred_full)[:, 0]
    else:
        y_val_pred_rescaled = y_val_pred_full[:, 0]

    y_val_true_full = np.zeros((len(y_val_true), len(features)))
    y_val_true_full[:, 0] = y_val_true
    y_val_true_rescaled = y_val_true_full[:, 0]

    plt.plot(dates_all, y_val_true_rescaled, label='Actual Prices (Validation)', color='green', linestyle='-', marker='o', linewidth=2)
    plt.plot(validation_dates, y_val_pred_rescaled, label='Predicted Prices (Validation)', color='orange', linestyle='--', marker='x', linewidth=2)

    if prediction_dates is not None and y_pred is not None:
        plt.plot(prediction_dates, y_pred, label='Predicted Prices (Next Days)', color='blue', linestyle='--', marker='s', linewidth=2)

    if specific_date is not None and merged_data is not None:
        future_actual_data = merged_data[merged_data['date'] >= specific_date]
        plt.plot(future_actual_data['date'], future_actual_data['y'], label='Actual Prices (After Prediction Date)', color='red', linestyle=':', marker='d', linewidth=2)

    plt.title('Stock Price: Actual vs Predicted', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price in USD', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(loc='best')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()











import plotly.graph_objects as go

def plot_results_interactive(validation_dates, dates_all, y_val_pred, y_val_true, prediction_dates=None, y_pred=None, merged_data=None, specific_date=None, scaler=None):
    # Create an empty figure object
    fig = go.Figure()

    # Create arrays for predicted and actual values
    features = ['y', 'Total Revenue', 'Net Income', 'Stockholders Equity', 'Total Liabilities', 'Total Assets']

    y_val_pred_full = np.zeros((len(y_val_pred), len(features)))
    y_val_pred_full[:, 0] = y_val_pred

    # Inverse transform if scaler is provided
    if scaler:
        y_val_pred_rescaled = scaler.inverse_transform(y_val_pred_full)[:, 0]
    else:
        y_val_pred_rescaled = y_val_pred_full[:, 0]

    y_val_true_full = np.zeros((len(y_val_true), len(features)))
    y_val_true_full[:, 0] = y_val_true
    y_val_true_rescaled = y_val_true_full[:, 0]

    # Plot the actual stock prices (validation)
    fig.add_trace(go.Scatter(
        x=dates_all,
        y=y_val_true_rescaled,
        mode='lines+markers',
        name='Actual Prices (Validation)',
        line=dict(color='green', width=2),
        marker=dict(symbol='circle', size=8)
    ))

    # Plot the predicted stock prices (validation)
    fig.add_trace(go.Scatter(
        x=validation_dates,
        y=y_val_pred_rescaled,
        mode='lines+markers',
        name='Predicted Prices (Validation)',
        line=dict(color='orange', dash='dash', width=2),
        marker=dict(symbol='x', size=8)
    ))

    # Plot the predicted future stock prices (if available)
    if prediction_dates is not None and y_pred is not None:
        fig.add_trace(go.Scatter(
            x=prediction_dates,
            y=y_pred,
            mode='lines+markers',
            name='Predicted Prices (Next Days)',
            line=dict(color='blue', dash='dot', width=2),
            marker=dict(symbol='square', size=8)
        ))

    # Plot the actual stock prices after the prediction date
    if specific_date is not None and merged_data is not None:
        future_actual_data = merged_data[merged_data['date'] >= specific_date]
        fig.add_trace(go.Scatter(
            x=future_actual_data['date'],
            y=future_actual_data['y'],
            mode='lines+markers',
            name='Actual Prices (After Prediction Date)',
            line=dict(color='red', dash='dot', width=2),
            marker=dict(symbol='diamond', size=8)
        ))

    # Set plot title and labels
    fig.update_layout(
        title='Stock Price: Actual vs Predicted',
        xaxis_title='Date',
        yaxis_title='Price in USD',
        legend=dict(y=1, x=0),
        hovermode="x unified"
    )

    # Show the plot
    fig.show()
