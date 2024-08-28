import yfinance as yf
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from alphavantage_api import get_days_elapsed_since_dividends
from election_utils import get_election_dates, convert_to_month_tuples, get_days_elapsed_for_country
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Load the stock data
symbol = "AAPL"
api_key = 'F1CSEY3K0H1XDJKO'
DATA_PATH = f"C:/Users/play4/Market_Project/{symbol}_data.json"

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

# Preparing the data
data = stock_hist[["Close", "Volume", "Open", "High", "Low"]]
data = data.rename(columns = {'Close':'Actual_Close'})  # Renaming Close to Actual_Close
data["Target"] = data["Actual_Close"].shift(-1) > data["Actual_Close"]  # Adjusting Target calculation
data.dropna(inplace=True)
stock_prev = data.shift(1)  # Shifted data for predictors

### Criterias for the training
election_data = get_election_dates()
month_tuples = convert_to_month_tuples(election_data)
days_elapsed = get_days_elapsed_for_country(country, month_tuples)

data["elections_trend"] = 0

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

time_elapsed_div = get_days_elapsed_since_dividends(symbol, api_key)
data["div_periode"] = 0
div_days = 15  # The interval around the dividend date
for nb_days_elapsed in time_elapsed_div:
    if nb_days_elapsed > div_days:
        start_pos = -nb_days_elapsed - div_days
        end_pos = -nb_days_elapsed + div_days + 1
        data.iloc[start_pos:end_pos, data.columns.get_loc('div_periode')] = 1

data["elections_trend"] = np.where(data['elections_trend'] == 1, data['Target'], 0)
data["div_trend"] = np.where(data['div_periode'] == 1, data['Target'], 0)

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

# Join previous day’s data
data = data.join(stock_prev, rsuffix="_prev").iloc[1:]

# Check for NaNs and fill them
data.fillna(data.mean(), inplace=True)

# List of predictors
predictors = ["Actual_Close", "Volume", "Open", "High", "Low", "weekly_mean", 
                "quarterly_mean", "annual_mean", 
                "annual_weekly_mean", "annual_quarterly_mean", 
                "open_close_ratio","high_close_ratio","low_close_ratio",
                "weekly_trend","monthly_trend","annual_trend","div_trend",
                "elections_trend"]

# Backtesting to improve our model

def back_test(data, model, predictors, start = 1000, step = 400):
    # A model is trained every "step" rows
    
    predictions = []
    
    for i in range(1000, data.shape[0], step):
        
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        model.fit(train[predictors], train["Target"])
        preds = model.predict_proba(test[predictors])[:,1] # Allow us to have values between 0 and 1 instead of 0/1
        preds = pd.Series(preds, index=test.index)
        
        preds[preds > .6] = 1 # We put the treshold here at 0.6, so it predicts it will go up with more confidence
        preds[preds<=.6] = 0
        
        combined = pd.concat({"Target": test["Target"], "Predictions": preds}, axis=1)
        
        predictions.append(combined)

    return pd.concat(predictions)


# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),  
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# 1. Without Backtesting: Evaluate on static split

train = data.iloc[:-100] 
test = data.iloc[-100:] 

# Define your custom interval
a, b = 0, 100  # Example values, can be adjusted

print(f"\nWithout Backtesting: Performance on Interval [{a}, {b}]")
precision_scores = []
recall_scores = []
f1_scores = []
roc_auc_scores = []
model_names = []

for model_name, model in models.items():
    # Train the model on the training data
    model.fit(train[predictors], train["Target"])
    
    # Predict on the test data
    preds = model.predict(test[predictors])
    preds_proba = model.predict_proba(test[predictors])[:, 1]
    
    # Create a DataFrame for easier manipulation and plotting
    results = pd.DataFrame({
        "Target": test["Target"],
        "Predictions": preds,
        "Prediction_Probabilities": preds_proba
    }, index=test.index)
    
    # Select the custom interval [a, b] from the results
    interval_predictions = results.iloc[a:b]
    
    # Calculate performance metrics for the selected interval
    precision_interval = precision_score(interval_predictions["Target"], interval_predictions["Predictions"])
    recall_interval = recall_score(interval_predictions["Target"], interval_predictions["Predictions"])
    f1_interval = f1_score(interval_predictions["Target"], interval_predictions["Predictions"])
    roc_auc_interval = roc_auc_score(interval_predictions["Target"], interval_predictions["Prediction_Probabilities"])
    
    print(f"\nPerformance metrics for {model_name} on the interval [{a}, {b}] (Without Backtesting):")
    print(f"Precision: {precision_interval:.4f}")
    print(f"Recall: {recall_interval:.4f}")
    print(f"F1-Score: {f1_interval:.4f}")
    print(f"ROC-AUC Score: {roc_auc_interval:.4f}")
    
    # Count the number of trades (predictions)
    trades = interval_predictions["Predictions"].value_counts()
    print(f"Number of trades for {model_name} on interval [{a}, {b}]:\n{trades}")

    # Plot target vs predictions for the interval
    plt.figure(figsize=(10, 6))
    plt.plot(interval_predictions.index, interval_predictions["Target"], label="Actual Target", color="blue", marker="o")
    plt.plot(interval_predictions.index, interval_predictions["Predictions"], label="Predicted", color="red", marker="x")
    plt.title(f"{model_name} - Target vs Predictions (Interval [{a}, {b}], Without Backtesting)")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.legend()
    plt.show()


# With Backtesting: Evaluate on the custom interval
print(f"\nWith Backtesting: Performance on Interval [{a}, {b}]")
for model_name, model in models.items():
    predictions = back_test(data.iloc[365:], model, predictors)
    
    # Select the custom interval [a, b] from the predictions
    interval_predictions = predictions.iloc[a:b]
    
    # Calculate performance metrics for the interval
    precision_interval = precision_score(interval_predictions["Target"], interval_predictions["Predictions"])
    recall_interval = recall_score(interval_predictions["Target"], interval_predictions["Predictions"])
    f1_interval = f1_score(interval_predictions["Target"], interval_predictions["Predictions"])
    roc_auc_interval = roc_auc_score(interval_predictions["Target"], interval_predictions["Predictions"])
    
    print(f"\nPerformance metrics for {model_name} on the interval [{a}, {b}] (With Backtesting):")
    print(f"Precision: {precision_interval:.4f}")
    print(f"Recall: {recall_interval:.4f}")
    print(f"F1-Score: {f1_interval:.4f}")
    print(f"ROC-AUC Score: {roc_auc_interval:.4f}")
    
    # Count the number of trades (predictions)
    trades = interval_predictions["Predictions"].value_counts()
    print(f"Number of trades for {model_name} on interval [{a}, {b}]:\n{trades}")

    # Plot target vs predictions for the interval
    plt.figure(figsize=(10, 6))
    plt.plot(interval_predictions.index, interval_predictions["Target"], label="Actual Target", color="blue", marker="o")
    plt.plot(interval_predictions.index, interval_predictions["Predictions"], label="Predicted", color="red", marker="x")
    plt.title(f"{model_name} - Target vs Predictions (Interval [{a}, {b}], With Backtesting)")
    plt.xlabel("Date")
    plt.ylabel("Target")
    plt.legend()
    plt.show()
