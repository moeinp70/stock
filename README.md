# Stock Price Prediction with Financial Data and LSTM Model

This project predicts stock prices using historical financial data and LSTM (Long Short-Term Memory) neural networks. It integrates data from various sources, such as Yahoo Finance and Finnhub, to gather historical stock prices and financial reports. The model trains on this data to provide accurate predictions for future stock prices.
## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Setup](#setup)
  - [Requirements](#requirements)
  - [Installation](#installation)
- [Usage](#usage)
  - [Data Fetching](#data-fetching)
  - [Model Training](#model-training)
  - [Prediction](#prediction)
- [Results and Visualization](#results-and-visualization)
- [Directory Structure](#directory-structure)
- [License](#license)
## Project Overview

This project leverages financial data to predict stock prices for specific stock tickers. The core of the project is an LSTM model that uses both stock price history and key financial metrics (e.g., revenue, net income, assets) to forecast future prices.

## Features
- Fetch historical stock prices and financial data from Yahoo Finance and Finnhub API.
- Perform sentiment analysis on news headlines related to the selected stock (optional).
- Train an LSTM model using historical data to predict future stock prices.
- Save and reuse trained models to avoid retraining on each run.
- Visualize actual vs. predicted prices interactively.
## Setup

### Requirements
The project requires Python 3.7 or above. Please install the dependencies listed in `requirements.txt`.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone <repository_url>
   cd stock-price-prediction
2. **Install Dependencies: Install the required packages by running:**:
   ```bash
    pip install -r requirements.txt
3. **API Keys:**:
   ```bash
    Get your API keys from Finnhub and Yahoo Finance.
   Add these keys in data_fetching.py, replacing the placeholder values.



---

## Usage



### Data Fetching
To fetch and preprocess financial data, run `data_fetching.py`. It collects stock prices and financial metrics for the specified stock ticker.

from data_fetching import get_stock_data, get_financial_data

stock_data = get_stock_data('MSFT')
financial_data = get_financial_data('MSFT')




### Model Training
Train the LSTM model using `model_training.py`. It supports both training from scratch and fine-tuning an existing model. Each model is saved for future use.

from model_training import train_lstm_model

model = train_lstm_model(X_train, y_train, window_size=60, prediction_horizon=30, epochs=50)




### Prediction
Use the trained model to predict future stock prices.

from model_training import predict_from_date

predicted_prices = predict_from_date(model, stock_data, specific_date='2024-10-13', prediction_horizon=30)


---

### Results and Visualization

The project provides interactive plots to compare actual vs. predicted stock prices. Use the `plot_results` function in `model_training.py` to visualize results.


from model_training import plot_results

plot_results(validation_dates, dates_all, y_val_pred, y_val_true, prediction_dates=prediction_dates, y_pred=y_pred)


## The visualization includes:

Actual stock prices during the training period.
Predicted stock prices for validation and future days.
Easy-to-understand color-coded lines for each dataset.



---

### Directory Structure


The project files are structured as follows:

```plaintext
.
├── data_fetching.py       # Scripts for fetching financial data
├── model_training.py      # LSTM model training and prediction functions
├── main.py                # Main script to run the entire workflow
├── requirements.txt       # Dependencies for the project
└── README.md              # Documentation file



---

### 8. License

This project is licensed under the MIT License.
