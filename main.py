from data_fetching import get_stock_data_with_cache, get_financial_data_with_cache
from model_training import create_supervised_dataset, train_lstm_model, predict_from_date, plot_results,plot_results_interactive
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from datetime import datetime
import pandas as pd
import random
import numpy as np
from tensorflow.keras.models import load_model
import os
import tensorflow as tf
# Function to save the trained model
# Function to save the trained model
def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved at {model_path}")

# Function to load the saved model
def load_saved_model(model_path):
    try:
        model = load_model(model_path)
        print(f"Model loaded from {model_path}")
        return model
    except:
        print(f"No saved model found at {model_path}, starting from scratch.")
        return None
# Merge Stock Data and Financial Data
def merge_data(stock_data, financial_data):
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    financial_data['date'] = pd.to_datetime(financial_data['date'])

    # Merge stock data with financial data using 'merge_asof' for aligning the quarterly data with daily stock data
    merged_data = pd.merge_asof(stock_data.sort_values('date'),
                                financial_data.sort_values('date'),
                                on='date', direction='backward')

    # Fill missing values in financial data with 0
    merged_data.fillna(0, inplace=True)

    return merged_data

# Set the seed for reproducibility
def set_random_seeds(seed_value=42):
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)
    random.seed(seed_value)


if __name__ == "__main__":
    set_random_seeds(42)
    stock_ticker = 'APA'
    specific_prediction_date = '2024-10-28'
    start_date = '2011-01-01'

    # Fetch historical stock data
    # Use the cached version of the stock and financial data fetching functions
    stock_data = get_stock_data_with_cache(stock_ticker)
    financial_data = get_financial_data_with_cache(stock_ticker)

    # Merge stock data with financial data
    merged_data = merge_data(stock_data, financial_data)

    # Create supervised dataset
    window_size = 150
    prediction_horizon = 22

    # You can change this if you don't want to normalize
    use_scaler = True
    scaler = StandardScaler() if use_scaler else None

    X, y, validation_dates, dates_all, y_true = create_supervised_dataset(merged_data, specific_prediction_date, window_size, prediction_horizon, scaler)
    # Generate a unique model path for each stock ticker
    model_dir = "saved_models"
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # Model path for saving/loading
    model_path = os.path.join(model_dir, f"lstm_model_{stock_ticker}.h5")

    # Check if a model already exists
    model = load_saved_model(model_path)

    if model is None:
        # Train the LSTM model if no saved model is found
        print(f"No saved model found for {stock_ticker}. Training from scratch.")
        model = train_lstm_model(X, y, window_size, prediction_horizon, epochs=100)

        # Save the model specific to this stock ticker
        save_model(model, model_path)
    else:
        # Fine-tune the existing model
        print(f"Saved model found for {stock_ticker}. Fine-tuning the model.")
        model = train_lstm_model(X, y, window_size, prediction_horizon, model=model, epochs=50)

        # Save the fine-tuned model
        save_model(model, model_path)

    # Train the LSTM model
    train_predictions = model.predict(X)

    # Predict future stock prices
    prediction_dates, y_pred = predict_from_date(model, merged_data, specific_prediction_date, prediction_horizon, window_size, scaler)

    # Plot results
    y_val_pred =  train_predictions[:-1,0]
    y_val_pred = np.append(y_val_pred, train_predictions[-1, :])
    y_val_true = y_true[:]
    plot_results_interactive(validation_dates, dates_all, y_val_pred, y_val_true, prediction_dates=prediction_dates, y_pred=y_pred, merged_data=merged_data, specific_date=specific_prediction_date, scaler=scaler)
