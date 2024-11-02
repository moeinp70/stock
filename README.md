Stock Price Prediction with LSTM and Financial Data
This repository contains a stock price prediction model built using LSTM neural networks. It leverages stock price history, financial data, and a fine-tuning capability to predict future prices.

Features
Financial Data Integration: Uses various financial indicators such as revenue, net income, stockholder equity, and more.
LSTM Model: A deep learning model capable of capturing sequential dependencies.
Fine-Tuning: Allows fine-tuning an already trained model for better performance without starting from scratch.
Data Caching: Stores fetched data to prevent redundant API calls, optimizing resources.
Interactive Plots: Provides interactive plots using Plotly for better visualization and analysis.
Project Structure
bash
Copy code
.
├── data_fetching.py         # Handles data fetching for stock price and financial data
├── model_training.py        # Defines LSTM model, training, and fine-tuning functionality
├── main.py                  # Main script to orchestrate fetching, training, and prediction
├── requirements.txt         # Python dependencies
└── README.md                # Project documentation
Requirements
Python 3.7 or later
TensorFlow
Pandas
Matplotlib
Plotly
Requests
finnhub-python (for accessing financial data via Finnhub)
Scikit-learn
To install all requirements:

bash
Copy code
pip install -r requirements.txt
Contents of requirements.txt:

Copy code
tensorflow
pandas
matplotlib
plotly
requests
finnhub-python
scikit-learn
yfinance
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/stock-price-prediction.git
cd stock-price-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Set up API keys:

Replace the placeholder API keys in data_fetching.py with your own Finnhub API key.
Sign up for a free Finnhub API key at Finnhub.io.
Usage
Run the main script:

To train and predict stock prices, execute:

bash
Copy code
python main.py
By default, the model saves trained weights and data for each stock ticker.

Customizing Configurations:

Adjust parameters such as window_size, prediction_horizon, epochs, and others in main.py and model_training.py to optimize for specific use cases.
Modify the stock_ticker variable in main.py to predict different stocks.
Using Interactive Plots:

The results will be displayed using interactive Plotly graphs.
Ensure that you have a web browser or an environment that supports interactive plots.
Explanation of Key Files
data_fetching.py: Contains functions to fetch and preprocess financial data and stock price data. Implements caching to save data locally and reduce redundant API calls.

model_training.py: Builds, trains, and fine-tunes the LSTM model. Supports loading existing models for fine-tuning and includes functions for creating supervised datasets and plotting results with Plotly.

main.py: Entry point that integrates data fetching, model training, prediction, and visualization.

Example Results
After running the script, results are visualized using interactive Plotly graphs, showing the actual vs. predicted stock prices. Fine-tuning the model helps stabilize predictions for subsequent runs.

Contributing
Fork the repository.

Create a new branch:

bash
Copy code
git checkout -b feature-branch
Make your changes and commit:

bash
Copy code
git commit -am 'Add new feature'
Push to the branch:

bash
Copy code
git push origin feature-branch
Create a new Pull Request.

License
This project is licensed under the MIT License.
