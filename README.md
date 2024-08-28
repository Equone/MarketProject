# Market Project

This repository contains code and resources for predicting stock prices using machine learning models. The project explores various models and techniques, with a focus on backtesting and feature engineering. We will explore different machine learning models, their limitations, and various ways of measuring the performance of a trained model.

_NB: I know there is no algorithm that can predict the future (and it would not be as simple as this). This project is for education purposes only.
This project was inspired by an article (I will update this README.md for the credits)._

## Project Overview

- **Data Source**: The data used in this project is historical stock data for Apple Inc. (`AAPL`), sourced from Yahoo Finance, but any company can be used.
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
We will see how it impacts the training : their nature, their number..

## An important point : Backtesting

```python
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
```

Backtesting simulates how a predictive model or trading strategy would have performed on historical data, providing insight into its potential future performance.

In the context of machine learning, backtesting involves retraining a model multiple times on different subsets of data to evaluate how the model would have performed in a real-world scenario, where data arrives sequentially over time. This approach is especially useful for time series data, where the order of data points matters, and traditional cross-validation might not be appropriate.

## Model Initialization

```python
# Initialize models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1),
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),  
    "KNN": KNeighborsClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}
```
This is the different models we will be using. Each comes with different pros and cons. We will talk about those in the **Results** section. We use a _dictionnary_ to simplify the code.

## Without BackTesting :

This section trains the models on a static train/test split and evaluates their performance; results will be discussed in the next part.

The following code train all models,test them on the wanted period and print all the metrics :

```python
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
```

And to see all the metrics with a better look :

```python
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
```

To add the Backtesting function, we just need to modify this :
```python
predictions = back_test(data.iloc[365:], model, predictors)
```
and the rest does not change.

# Results
In this part, we look at the different results we have from the company Apple (symbol: `AAPL`), the _25 august 2024_.

_If you need explanations about the metrics and notions that are used in this part, they are well detailled in this article :_ https://towardsdatascience.com/a-look-at-precision-recall-and-f1-score-36b5fd0dd3ec

## All metrics
Here are the results I obtained from my tests :

### _Without BackTesting :_
![AAPL_NoBackTesting](https://github.com/user-attachments/assets/38168921-d62a-4447-90df-ed9c0863d8d2)

### _With BackTesting :_
![AAPL_NoBackTesting](https://github.com/user-attachments/assets/65a5f26b-bab1-40b2-8593-2d8acd305194)

This table sums up the impact of the addition of BackTesting on our models.

|Model \ Metric| Precision  | Recall | F1-Score | ROC-AUC |
| ------------- | ------------- |  ------------- |  ------------- |  ------------- |
|Gradient Boosting  | **Increase**  | **Increase**  | **Increase**  | **Increase**  |
| KNN  | _Decrease_  | _Decrease_  | _Decrease_  | _Decrease_  |
| SVM  | N/A  | N/A  | N/A  | _Decrease_  |
| Logisitic Regression  | **Increase**  | _Decrease_  | _Decrease_  | Constant  |
| Random Forest  | Constant  | **Increase**  | **Increase**  | **Increase**  |

Looking at the impact of the introduction of Backtesting on metrics tells us how our models react to more dynamic data. As an example, the KNN model is taking a huge hit, while Random Forest takes a lot of profits from it. 

It is **relevant** to look at the impact of BackTesting, as in some fields - like finances - data are very variable.

Additionnaly, we can look at this graph, which shows the ratio between _True Positive Rate_ and _False Positive Rate_ (without BackTesting here):

![all](https://github.com/user-attachments/assets/70d7ab0e-a7a6-4806-a781-62d28923286e)

As the table already suggested, **Random Forest** and **Gradient Boosting** are strong at this task; while **Logistic Regression** or **SVM** perform weaker, which may show that they are not well-suited for this time-series prediction task.

## Tests of the models on the last 100 days

Once the models are trained, we apply them on the last 100 days in order to see how good they are, and how many trades we would have made in this period.

__NB: While the code analyzes the last 100 days, you can adjust this timeframe. To focus on specific event dates, locate the appropriate index and exclude that period from model training. Furthermore, every graph can be created with the code, I only took some to write this README.

For every model, we plot the _Target_ and the _Prediction_ by the date, in order to know if we were right or not.
Let's take some **relevant** examples, so we can see some limits of machine learning.

### Models not adapted for the data

**Logistic Regression**, without BackTesting :

![Figure 2024-08-27 121835 (1)](https://github.com/user-attachments/assets/943ac09a-d516-44e7-b224-7d143105cbbd)

**SVM**, without BackTesting :

![Figure 2024-08-27 121835 (2)](https://github.com/user-attachments/assets/b214381f-cd8a-4292-9be8-7f46897bf84b)

As we can see, it is quite polarized. For **Logistic Regression**, we would have made **100** trades for a global precision of _.65_ - which does not mean profit, as we do not consider the ratio `open\close`.

Obviously, this would not be a good strategy at all and therefore, this kind of model for this data and without BackTesting, is not viable nor reliable.

### Models adapted for the data

**Gradient Boosting**, without BackTesting :

![Figure 2024-08-27 121835 (3)](https://github.com/user-attachments/assets/c161e260-d079-4e1c-810c-95c14dea08c5)

**KNN**, without BackTesting :

![Figure 2024-08-27 121835 (3)](https://github.com/user-attachments/assets/7fdb736c-acc1-4793-8b2a-46c2bc4f8a0a)

Respectively, with **Gradient Boosting** and **KNN** we would have made **11** and **47** trades. However, metrics are quite different :

|Model \ Metric| Precision  | Recall | F1-Score | ROC-AUC |
| ------------- | ------------- |  ------------- |  ------------- |  ------------- |
|Gradient Boosting  | 0.6364 | 0.1077 | 0.1842 | 0.6055 |
| KNN  | 0.6981 | 0.5692 | 0.6271 | 0.5622 |

This correspond to two different strategies : a "_safe_" one, and a more "_aggressive_" one, depending on how we want to approach our investments and how much risks we are willing to take. However, **KNN** metrics are still relatively high compared to its rival.

You can still play on different approachs, parameters to see how models react, etc. I just put some examples here as a sample, we can do much more in-depth analysis with this.

## Limits

### Overfitting

The reason why the models above were without BackTesting is : the addition of BackTesting + many many criterias may lead to overfitting.

Here is a clear example : 

**Random Forest**, with BackTesting :

![Figure 2024-08-27 121835 (5)](https://github.com/user-attachments/assets/319cb744-6c97-4a1a-a94b-10254bcbefa8)

with really (really) high metrics :

![image](https://github.com/user-attachments/assets/f9f10586-7695-4d82-91bd-508ae133543c)


Overfitting means the model learned _too much_ the data -  not only the underlying patterns in the training data but also the noise and random fluctuations - this can lead to several issues:

1. Poor generalization : weak perfomances on new (unseen) data.
2. High variance : this is linked to the point before; often we want things to be as stable as possible.
3. Loss of robustness : undesired patterns can be reproduced on new data.
4. Misleading Metrics : a lack of vigilance on this phenomenon can be enhanced by the fact that metrics are good in this case.

Nevertheless, solutions exist : More data, early stopping, simplification of the model (like removing some criterias),...

If you want to see different behaviours, feel free to play with criterias, BackTesting parameters, etc.

## Possible improvements

Of course, some criterias may be added or changed. Furthermore, we could add a feature that tells us how much we would have won if trusting a model. Also, removing some parts of the training data could be an idea to reduces thz variance.

# Conclusion

Through this project, we saw the process of training different machine learning model on stock prices, the impact of implementing features like BackTesting and some of their limitations (overfitting).

***Keys points :***
- **Importance of criterias**, as it plays a crucial role in the performance of the models.
- **Role of BackTesting**, which is essential for evaluating a model's ability to generalize to unseen data.
- **Model comparison**
- **Understanding Overfitting** and its implications
- **Using various techniques to get informations / data :** Webscrapping, API, specific librairies

***Final Thoughts:***
Predicting stock prices is a complex and uncertain task, as financial markets are influenced by a wide range of factors, many of which are impossible to quantify. While machine learning offers powerful tools for analyzing historical data and making predictions, it is important to approach such tasks with caution and recognize the limitations of these models.

This project is not intended for real-world financial applications; it is here to give me an idea of the process of data preparation, feature engineering, model training, and evaluation, providing a foundation for further exploration and experimentation in the field of financial modeling. And as we say in France, for legal reasons : "_Les performances passées ne préjugent pas des performances futures_"

