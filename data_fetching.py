import requests
import yfinance as yf
import pandas as pd
import finnhub
import re

api_key = 'cs6h7kpr01qkeuli3000cs6h7kpr01qkeuli300g'
finnhub_client = finnhub.Client(api_key="cs6h7kpr01qkeuli3000cs6h7kpr01qkeuli300g")

import os
import pandas as pd
from datetime import datetime


# Function to cache stock data
def cache_stock_data(stock_data, filename):
    stock_data.to_csv(filename, index=False)


# Function to load cached stock data if available
def load_cached_data(filename):
    if os.path.exists(filename):
        print(f"Loading cached data from {filename}")
        return pd.read_csv(filename)
    return None


# Function to get stock data with caching
def get_stock_data_with_cache(stock_ticker, cache_dir="cache"):
    today = datetime.today().strftime('%Y-%m-%d')
    stock_cache_file = os.path.join(cache_dir, f"{stock_ticker}_stock_{today}.csv")

    # Check if data for today is already cached
    stock_data = load_cached_data(stock_cache_file)
    if stock_data is not None:
        return stock_data

    # Fetch stock data from API
    stock_data = get_stock_data(stock_ticker)

    # Cache the data for future use
    os.makedirs(cache_dir, exist_ok=True)
    cache_stock_data(stock_data, stock_cache_file)

    return stock_data


# Similar caching can be applied for financial data
def get_financial_data_with_cache(stock_ticker, cache_dir="cache"):
    today = datetime.today().strftime('%Y-%m-%d')
    financial_cache_file = os.path.join(cache_dir, f"{stock_ticker}_financial_{today}.csv")

    # Check if data for today is already cached
    financial_data = load_cached_data(financial_cache_file)
    if financial_data is not None:
        return financial_data

    # Fetch financial data from API
    financial_data = get_financial_data(stock_ticker)

    # Cache the data for future use
    os.makedirs(cache_dir, exist_ok=True)
    cache_stock_data(financial_data, financial_cache_file)

    return financial_data

# Fetch Stock Data
def get_stock_data(stock_ticker):
    stock_data = yf.download(stock_ticker, start="2011-01-01", end=pd.Timestamp.today().strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Close']].rename(columns={'Date': 'date', 'Close': 'y'})
    return stock_data


# Fetch Financial Data for a Stock Ticker from Finnhub API
# Utility function for fuzzy matching
def find_value_by_keyword(dictionary, keyword):
    """
    Search for a keyword in a dictionary's keys and return the corresponding value.
    """
    for key, value in dictionary.items():
        if re.search(keyword, key, re.IGNORECASE):  # Case-insensitive search for the keyword in key
            return value
    return 0  # Default value if no match is found

# Fetch Financial Data for a Stock Ticker using Finnhub
def get_financial_data(stock_ticker):
    try:
        # Fetch historical quarterly financial statements
        financial_response = finnhub_client.financials_reported(symbol=stock_ticker, freq='quarterly')

        # Check if response contains data
        if 'data' not in financial_response or not financial_response['data']:
            print(f"No quarterly reports found for {stock_ticker}")
            return pd.DataFrame()

        # Extract financials from the response
        quarterly_reports = financial_response['data']

        financial_records = []
        for report in quarterly_reports:
            # Extract the report date
            report_date = report.get('endDate')

            # Extract income statement (ic) and balance sheet (bs)
            ic = report.get('report', {}).get('ic', [])  # Income statement items
            bs = report.get('report', {}).get('bs', [])  # Balance sheet items

            # Convert lists to dictionaries for easier extraction
            ic_dict = {item['label']: item['value'] for item in ic}  # Income statement dictionary
            bs_dict = {item['label']: item['value'] for item in bs}  # Balance sheet dictionary

            # Use fuzzy matching to find values for relevant financial metrics
            total_revenue = find_value_by_keyword(ic_dict, 'Revenue')
            net_income = find_value_by_keyword(ic_dict, 'Net income')
            total_assets = find_value_by_keyword(bs_dict, 'Assets')
            total_liabilities = find_value_by_keyword(bs_dict, 'Liabilities')
            stockholders_equity = find_value_by_keyword(bs_dict, 'Equity')

            # Append extracted data into a dictionary using relevant financial labels
            financial_records.append({
                'date': report_date,
                'Total Revenue': total_revenue,
                'Net Income': net_income,
                'Total Assets': total_assets,
                'Total Liabilities': total_liabilities,
                'Stockholders Equity': stockholders_equity
            })

        # Convert the financial_records into a DataFrame
        financial_df = pd.DataFrame(financial_records)
        financial_df['date'] = pd.to_datetime(financial_df['date'])

        return financial_df

    except Exception as e:
        print(f"Exception occurred while fetching financial data: {str(e)}")
        return pd.DataFrame()


# Fetch News Data using Finnhub API
# Fetch News Data from Finnhub API
# Fetch News Data using Finnhub API
def get_stock_news(stock_ticker, start_date, end_date):
    # API endpoint for company news
    news_url = f'https://finnhub.io/api/v1/company-news?symbol={stock_ticker}&from={start_date}&to={end_date}&token={api_key}'

    try:
        # Fetch news data
        news_response = requests.get(news_url)

        if news_response.status_code == 200:
            news_data = news_response.json()

            # Process and extract relevant fields for sentiment analysis
            news_list = []
            for article in news_data:
                news_list.append({
                    'date': article.get('datetime', '')[:10],  # Use only the date part
                    'headline': article.get('headline', ''),
                    'source': article.get('source', ''),
                    'summary': article.get('summary', '')
                })

            news_df = pd.DataFrame(news_list)
            return news_df
        else:
            print(f"Error fetching news data: {news_response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        print(f"Exception occurred while fetching news data: {str(e)}")
        return pd.DataFrame()


"""
# Fetch Financial Data for a Stock Ticker using Alpha Vantage
def get_financial_data(stock_ticker):


    # Fetch income statement
    income_statement_params = {
        "function": "INCOME_STATEMENT",
        "symbol": stock_ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    income_statement_response = requests.get(base_url, params=income_statement_params)
    income_statement_data = income_statement_response.json()

    # Fetch balance sheet
    balance_sheet_params = {
        "function": "BALANCE_SHEET",
        "symbol": stock_ticker,
        "apikey": ALPHA_VANTAGE_API_KEY
    }
    balance_sheet_response = requests.get(base_url, params=balance_sheet_params)
    balance_sheet_data = balance_sheet_response.json()

    # Extract relevant fields for quarterly data
    quarterly_income = income_statement_data.get("quarterlyReports", [])
    quarterly_balance_sheet = balance_sheet_data.get("quarterlyReports", [])

    if not quarterly_income or not quarterly_balance_sheet:
        print("No financial data available.")
        return pd.DataFrame()

    # Create a list of dictionaries for financial data, with fiscalDateEnding as the key
    income_data = {report['fiscalDateEnding']: report for report in quarterly_income}
    balance_sheet_data = {report['fiscalDateEnding']: report for report in quarterly_balance_sheet}

    # Get the intersection of available dates
    common_dates = set(income_data.keys()).intersection(set(balance_sheet_data.keys()))

    # Sort the dates
    common_dates = sorted(common_dates, reverse=True)

    # Build the DataFrame with aligned dates
    financial_data = []
    for date in common_dates:
        financial_data.append({
            'date': date,
            'Total Revenue': float(income_data[date].get('totalRevenue', 0)),
            'Net Income': float(income_data[date].get('netIncome', 0)),
            'Stockholders Equity': float(balance_sheet_data[date].get('totalShareholderEquity', 0)),
            'Total Liabilities': float(balance_sheet_data[date].get('totalLiabilities', 0)),
            'Total Assets': float(balance_sheet_data[date].get('totalAssets', 0)),
            'Net Debt': float(balance_sheet_data[date].get('netDebt', 0))
        })

    # Convert the list of dictionaries to a DataFrame
    financial_df = pd.DataFrame(financial_data)

    # Convert date column to datetime
    financial_df['date'] = pd.to_datetime(financial_df['date'])

    return financial_df


import requests
from datetime import datetime


# Fetch News Data for Sentiment Analysis using Alpha Vantage News API
def get_stock_news(stock_ticker, end_date):
    api_key = 'F1CNN4ILYJEI7JZD'  # Your Alpha Vantage API key
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={stock_ticker}&apikey={api_key}"

    response = requests.get(url)
    news_data = response.json()

    # Ensure the response contains news data
    if 'feed' not in news_data:
        print("No news articles found in the response.")
        return []

    articles = []
    for article in news_data['feed']:
        published_at = article.get('time_published', None)

        # Debugging: Print the published_at field to check its format
        print(f"Published at: {published_at}")

        if published_at:
            try:
                # Try parsing the date in 'YYYYMMDD' format (Alpha Vantage format)
                article_date = datetime.strptime(published_at[:8], "%Y%m%d")
                if article_date <= datetime.strptime(end_date, "%Y-%m-%d"):
                    articles.append({
                        'date': article_date.strftime('%Y-%m-%d'),
                        'title': article.get('title', ''),
                        'sentiment': article.get('overall_sentiment_score', 0)  # Use overall sentiment if available
                    })
            except ValueError as e:
                print(f"Error parsing date: {e}")

    return articles

"""
"""
import yfinance as yf
import requests
import pandas as pd
from datetime import datetime
from textblob import TextBlob

# Fetch Stock Data for a Given Ticker
def get_stock_data(stock_ticker, start_date="2017-01-01"):
    stock_data = yf.download(stock_ticker, start=start_date, end=datetime.today().strftime('%Y-%m-%d'))
    stock_data.reset_index(inplace=True)
    stock_data = stock_data[['Date', 'Close']]
    stock_data.rename(columns={'Date': 'date', 'Close': 'y'}, inplace=True)
    return stock_data

# Fetch Financial Data for a Stock Ticker
def get_financial_data(stock_ticker):
    stock = yf.Ticker(stock_ticker)
    quarterly_income = stock.quarterly_financials.T
    quarterly_balance_sheet = stock.quarterly_balance_sheet.T

    financial_data = pd.DataFrame({
        'date': quarterly_income.index,
        'Total Revenue': quarterly_income.get('Total Revenue', 0),
        'Net Income': quarterly_income.get('Net Income', 0),
        'Stockholders Equity': quarterly_balance_sheet.get('Stockholders Equity', 0),
        'Total Liabilities': quarterly_balance_sheet.get('Total Liabilities Net Minority Interest', 0),
        'Total Assets': quarterly_balance_sheet.get('Total Assets', 0),
        'Net Debt': quarterly_balance_sheet.get('Net Debt', 0)
    })

    financial_data.reset_index(drop=True, inplace=True)
    financial_data['date'] = pd.to_datetime(financial_data['date'])
    return financial_data

# Fetch News Data for Sentiment Analysis
def get_stock_news(stock_ticker, end_date):
    api_key = 'c3c00d1d5bd54c18a26765f3149be0ee'  # Replace with your API key
    url = f"https://newsapi.org/v2/everything?q={stock_ticker}&to={end_date}&apiKey={api_key}"
    response = requests.get(url)
    news_data = response.json()
    return news_data['articles']

# Perform Sentiment Analysis on News Headlines
def analyze_sentiment(news_articles):
    sentiments = []
    for article in news_articles:
        headline = article['title']
        date = article['publishedAt'][:10]
        sentiment_score = TextBlob(headline).sentiment.polarity
        sentiments.append({'date': date, 'sentiment': sentiment_score})
    sentiment_df = pd.DataFrame(sentiments)
    sentiment_df = sentiment_df.groupby('date').mean().reset_index()
    return sentiment_df

"""