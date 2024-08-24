# Market Project

This repository contains code and resources for predicting stock prices using machine learning models. The project explores various models and techniques, with a focus on backtesting and feature engineering. Of course, I know there is no algorithm that can predict the future; this project is only for an education purposes only.
This project was inspired by an article on dataset (I will update this README.md for the credits).

## Project Overview

- **Data Source**: The data used in this project is historical stock data for Apple Inc. (`AAPL`), sourced from Yahoo Finance.
- **Objective**: To predict whether the stock price will go up the next day based on various features derived from historical data.
- **Models**: We compare the performance of multiple machine learning models, with and without backtesting, and with varying sets of features.

## Repository Structure

- `data/`: Contains the stock data in JSON format.
- `src/`: Python scripts for data processing, model training, and evaluation.
- `results/`: Contains performance comparison plots and metrics.
- `README.md`: Project overview and instructions.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Equone/MarketProject.git
cd MarketProject
pip install -r requirements.txt
```

# Global overview of the code



## Imports and Setup
```python
import yfinance as yf
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
from alphavantage_api import get_days_elapsed_since_dividends
from election_utils import get_election_dates, convert_to_month_tuples, get_days_elapsed_for_country
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
```

If you looked at ```requirements.txt```, you should have every packages installed. 
```election_utils``` will be useful to web scrape data from Wikipedia, and ```alphavantage_api``` to have an access to data concerning companies.



## Loading and Preparing Data

```python
# Load the stock data
symbol = "AAPL"
api_key = 'F1CSEY3K0H1XDJKO' # Key for the alphavantage API.
DATA_PATH = f"C:/Users/user_name/Market_Project/{symbol}_data.json"

if os.path.exists(DATA_PATH):
    stock = yf.Ticker(symbol)
    info = stock.info
    country = info.get('country', 'Country not found')
    with open(DATA_PATH) as f:
        stock_hist = pd.read_json(DATA_PATH)
else:
    stock = yf.Ticker(symbol)
    stock_hist = stock.history(period="max")
    stock_hist.to_json(DATA_PATH)
    info = stock.info
    country = info.get('country', 'Country not found')
```
We check if we already imported data from yfinance in order to avoid useless requests.



## Criterias for the training
We will need multiple criterias in order to train our different models. Here are the 'basic' ones :

```python
weekly_trend = data["Target"].shift(1).rolling(7).sum()
monthly_trend = data["Target"].shift(1).rolling(30).sum()
annual_trend = data["Target"].shift(1).rolling(365).sum()

data["weekly_trend"] = weekly_trend
data["monthly_trend"] = monthly_trend
data["annual_trend"] = annual_trend

weekly_mean = data["Actual_Close"].rolling(7).mean()
quarterly_mean = data["Actual_Close"].rolling(90).mean()
annual_mean = data["Actual_Close"].rolling(365).mean()

data["weekly_mean"] = weekly_mean / data["Actual_Close"]
data["quarterly_mean"] = quarterly_mean / data["Actual_Close"]
data["annual_mean"] = annual_mean / data["Actual_Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]

data["open_close_ratio"] = data["Open"] / data["Actual_Close"]
data["high_close_ratio"] = data["High"] / data["Actual_Close"]
data["low_close_ratio"] = data["Low"] / data["Actual_Close"]
```

Names are clear enough.
Nevertheless, many things may impact economy. 
In order to improve my models, I chosed two supplementary things : dates where dividends were given and election dates.
It is where the ```election_utils.py``` and ```alphavantage_api.py``` play their role.


## election_utils.py file
The goal of this file is to extract the date of 
