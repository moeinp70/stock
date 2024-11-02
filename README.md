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

This project leverages financial data to predict stock prices for specific stock tickers. The core of the project is an LSTM model that uses both stock price history and key financial metrics (such as revenue, net income, and assets) to forecast future prices. The model is trained using historical data, which helps in creating more accurate predictions.

The project is modularized into separate files, making it easy to customize, update, and fine-tune specific components, such as data fetching or model parameters.


## Features

- **Data Collection**: Fetches historical stock prices and key financial data from Yahoo Finance and the Finnhub API.
- **Sentiment Analysis**: (Optional) Performs sentiment analysis on news headlines related to the selected stock ticker.
- **LSTM Model for Prediction**: Utilizes an LSTM neural network model to predict future stock prices based on historical data.
- **Model Persistence**: Saves trained models to allow for reuse, avoiding the need for retraining every time.
- **Interactive Visualization**: Provides interactive visualizations of actual vs. predicted stock prices, allowing for better insights.

  
## Setup

### Requirements
- Python 3.7 or above
- Install the dependencies listed in `requirements.txt`.

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

```python
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

**.
├── data_fetching.py       # Scripts for fetching financial data
├── model_training.py      # LSTM model training and prediction functions
├── main.py                # Main script to run the entire workflow
├── requirements.txt       # Dependencies for the project
└── README.md              # Documentation file
**


---

### 8. License

This project is licensed under the MIT License.
