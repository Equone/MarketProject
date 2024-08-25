# Market Project

This repository contains code and resources for predicting stock prices using machine learning models. The project explores various models and techniques, with a focus on backtesting and feature engineering. We will see the different machine learning models, their limits and the different ways of measuring the performances of a trained model.

_NB: I know there is no algorithm that can predict the future; and it would not be as simple as this. This project is only for an education purposes only.
This project was inspired by an article on dataset (I will update this README.md for the credits)._

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
The variable `symbol` can be changed into the one you are interested in. In this code, I have chosen Apple, `"AAPL"`.
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

They can be divided into multiple sub-sections :

### Trends
```python
weekly_trend = data["Target"].shift(1).rolling(7).sum()
monthly_trend = data["Target"].shift(1).rolling(30).sum()
annual_trend = data["Target"].shift(1).rolling(365).sum()

data["weekly_trend"] = weekly_trend
data["monthly_trend"] = monthly_trend
data["annual_trend"] = annual_trend
```

### Means
```python
weekly_mean = data["Actual_Close"].rolling(7).mean()
quarterly_mean = data["Actual_Close"].rolling(90).mean()
annual_mean = data["Actual_Close"].rolling(365).mean()

data["weekly_mean"] = weekly_mean / data["Actual_Close"]
data["quarterly_mean"] = quarterly_mean / data["Actual_Close"]
data["annual_mean"] = annual_mean / data["Actual_Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
```

### Ratios
```python
data["open_close_ratio"] = data["Open"] / data["Actual_Close"]
data["high_close_ratio"] = data["High"] / data["Actual_Close"]
data["low_close_ratio"] = data["Low"] / data["Actual_Close"]
```

Nevertheless, many things may impact economy. 
In order to improve my models, I chosed two supplementary things : dates where dividends were given and election dates.
It is where the ```election_utils.py``` and ```alphavantage_api.py``` play their role.


## election_utils.py file

The goal of this file is to extract the date of the different elections from the wikipedia page : https://en.wikipedia.org/wiki/List_of_next_general_elections

This script contains functions that handle election data, such as scraping the next general election dates, processing these dates, and calculating the days elapsed since these elections. The primary objective of this module is to create features that could be used for predictive modeling in stock price prediction tasks.

_NB: we calculate the days elapsed since the given election because in the dataframe **data** we use, days are counted since the introduction of the company into the stock market. Then, we assume elections are every 5 years. We can be more precise if we check, for each countries, precisely the period between elections. However, most of the time, 5 years is accurate._

### Functions Overview:
1. `get_election_dates()`:
 Scrapes election data from the Wikipedia page listing the next general elections. The data includes upcoming presidential and legislative elections, as well as the last election dates.

2. `extract_month(date_str)`:
 Extracts the month from a date string, which is useful for understanding when an election took place or is scheduled to occur. This helper function splits a date string and returns the month part, which is later used to calculate time-related features.

3. `convert_to_month_tuples(election_data)`:
 Converts the election data into tuples that contain the country, the month of the next legislative election, and the month of the last presidential election.

4. `days_since_month(month_str, year_str)`:
 Calculates the number of days elapsed since the specified month and year. All of these functions are here for one reason : the next function, and how we manage time in our dataframe.

5. `get_days_elapsed_for_country(country_name, month_tuples)`:
 For a given country, this function calculates the days elapsed since the last presidential or legislative election (that are present in the tuple). Then, when we have this value, we can assume that the election period was at : ```data.iloc[-days_elapsed - period/2 : -days_elapsed + period/2]``` where the value `period` correspond to the time where we consider elections having a relevant impact.

### Implementation
Here is how we proceed to integrate those functions :
```python
n = len(data)
years_interval = 5 * 365  # 5 years in days

for days in days_elapsed:
    if isinstance(days, int):  # Check if 'days' is an integer
        while True:
            pos_minus = max(-days - 15, 0)  # Ensure not to go before the start of the DataFrame
            pos_plus = min(-days + 15, n - 1)  # Ensure not to exceed the DataFrame bounds
            
            # Set 1s in the 'elections_trend' column within the calculated positions
            data.iloc[pos_minus:pos_plus, data.columns.get_loc("elections_trend")] = 1
            
            # Move to the previous 5-year period
            days += years_interval

            # Stop if the calculated position is beyond the DataFrame's range
            if days >= n:
                break
```
Elections are now considered in our models. The parameter that can be change is the `15`, which is the interval around the election.


## alphavantage_api.py file

This script provides functionality to retrieve and process dividend data for a specific stock symbol using the Alpha Vantage API. The primary function is used to calculate the days elapsed since the last dividend distributions, which is another relevant criteria in share prices.

### Functions Overview:
1. `get_days_elapsed_since_dividends(symbol, api_key)`:
 Fetches the weekly adjusted time series data for a given stock symbol and calculates the days elapsed since each non-zero dividend distribution. The reason why we calculate days elapsed is the same as before.

### Implementation
Here is how we proceed to integrate this script :
```python
time_elapsed_div = get_days_elapsed_since_dividends(symbol, api_key)
data["div_periode"] = 0 # 0 everywhere first
div_days = 15  # The interval around the dividend date

for nb_days_elapsed in time_elapsed_div:
    if nb_days_elapsed > div_days:
        start_pos = -nb_days_elapsed - div_days
        end_pos = -nb_days_elapsed + div_days + 1
        data.iloc[start_pos:end_pos, data.columns.get_loc('div_periode')] = 1
```
Thus, if the company gives dividends, this criteria will be considered into our training.


**Now, we have set up our criterias. We can begin the training of our different models.**

## Defining Predictors

This list defines the features that will be used as inputs for the models.
```python
# List of predictors
predictors = ["Actual_Close", "Volume", "Open", "High", "Low", "weekly_mean", 
                "quarterly_mean", "annual_mean", 
                "annual_weekly_mean", "annual_quarterly_mean", 
                "open_close_ratio","high_close_ratio","low_close_ratio",
                "weekly_trend","monthly_trend","annual_trend","div_trend",
                "elections_trend"]
```
