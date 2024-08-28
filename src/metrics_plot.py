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

print("\nWithout Backtesting:")
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
    
    precision = precision_score(test["Target"], preds)
    recall = recall_score(test["Target"], preds)
    f1 = f1_score(test["Target"], preds)
    roc_auc = roc_auc_score(test["Target"], preds_proba)
    
    precision_scores.append(precision)
    recall_scores.append(recall)
    f1_scores.append(f1)
    roc_auc_scores.append(roc_auc)
    model_names.append(model_name)
    
    print(f"\nPerformance metrics for {model_name}:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test["Target"], preds_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC = {roc_auc:.2f})')
    

# Plot diagonal line for reference (no skill classifier)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Adding labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - without Backtesting')
plt.legend(loc="lower right")

# Show the plot
plt.show()


# Plotting the metrics for static split (without backtesting)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.barh(model_names, precision_scores, color='blue')
plt.title('Precision (Without Backtesting)')

plt.subplot(2, 2, 2)
plt.barh(model_names, recall_scores, color='green')
plt.title('Recall (Without Backtesting)')

plt.subplot(2, 2, 3)
plt.barh(model_names, f1_scores, color='red')
plt.title('F1-Score (Without Backtesting)')

plt.subplot(2, 2, 4)
plt.barh(model_names, roc_auc_scores, color='purple')
plt.title('ROC-AUC (Without Backtesting)')

plt.tight_layout()
plt.show()


# 2. With Backtesting: Evaluate using backtesting approach
print("\nWith Backtesting:")
precision_backtest_scores = []
recall_backtest_scores = []
f1_backtest_scores = []
roc_auc_backtest_scores = []

for model_name, model in models.items():
    predictions = back_test(data.iloc[365:], model, predictors)
    
    precision_backtest = precision_score(predictions["Target"], predictions["Predictions"])
    recall_backtest = recall_score(predictions["Target"], predictions["Predictions"])
    f1_backtest = f1_score(predictions["Target"], predictions["Predictions"])
    roc_auc_backtest = roc_auc_score(predictions["Target"], predictions["Predictions"])
    
    precision_backtest_scores.append(precision_backtest)
    recall_backtest_scores.append(recall_backtest)
    f1_backtest_scores.append(f1_backtest)
    roc_auc_backtest_scores.append(roc_auc_backtest)
    
    print(f"\nBacktesting performance metrics for {model_name}:")
    print(f"Precision: {precision_backtest:.4f}")
    print(f"Recall: {recall_backtest:.4f}")
    print(f"F1-Score: {f1_backtest:.4f}")
    print(f"ROC-AUC Score: {roc_auc_backtest:.4f}")
    
    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(test["Target"], preds_proba)
    plt.plot(fpr, tpr, label=f'{model_name} (ROC-AUC = {roc_auc:.2f})')

    
# Plot diagonal line for reference (no skill classifier)
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

# Adding labels and title
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve - with Backtesting')
plt.legend(loc="lower right")

# Show the plot
plt.show()

# Plotting the metrics for backtesting
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.barh(model_names, precision_backtest_scores, color='blue')
plt.title('Precision (With Backtesting)')

plt.subplot(2, 2, 2)
plt.barh(model_names, recall_backtest_scores, color='green')
plt.title('Recall (With Backtesting)')

plt.subplot(2, 2, 3)
plt.barh(model_names, f1_backtest_scores, color='red')
plt.title('F1-Score (With Backtesting)')

plt.subplot(2, 2, 4)
plt.barh(model_names, roc_auc_backtest_scores, color='purple')
plt.title('ROC-AUC (With Backtesting)')

plt.tight_layout()
plt.show()

