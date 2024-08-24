# Stock Price Prediction

This repository contains code and resources for predicting stock prices using machine learning models. The project explores various models and techniques, with a focus on backtesting and feature engineering.

## Project Overview

- **Data Source**: The data used in this project is historical stock data for Apple Inc. (`AAPL`), sourced from Yahoo Finance.
- **Objective**: To predict whether the stock price will go up the next day based on various features derived from historical data.
- **Models**: We compare the performance of multiple machine learning models, with and without backtesting, and with varying sets of features.

## Repository Structure

- `data/`: Contains the stock data in JSON format.
- `notebooks/`: Jupyter notebooks for exploratory data analysis.
- `src/`: Python scripts for data processing, model training, and evaluation.
- `results/`: Contains performance comparison plots and metrics.
- `README.md`: Project overview and instructions.

## Installation

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/Equone/MarketProject.git
cd Stock_Price_Prediction
pip install -r requirements.txt
