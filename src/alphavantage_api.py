from datetime import datetime
import requests

"""
My API Key : get your API key on the alphavantage website
"""

def get_days_elapsed_since_dividends(symbol, api_key):
    """
    Retrieves the number of days that have elapsed since the dates of non-zero dividends for a given company.

    :param symbol: The stock symbol of the company (e.g., 'MSFT' for Microsoft).
    :param api_key: The API key to access Alpha Vantage.
    :return: A list of the number of days elapsed since each dividend date with non-zero amounts.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Loop through the data to find dates with non-zero dividends
    dates_with_dividends = []

    for date, metrics in data["Weekly Adjusted Time Series"].items():
        dividend_amount = float(metrics["7. dividend amount"])
        if dividend_amount > 0:
            dates_with_dividends.append((date, dividend_amount))

    # List of dividend dates
    dividend_dates = [dates_with_dividends[i][0] for i in range(len(dates_with_dividends))]
    time_elapsed_since_div = []

    # Get today's date
    today = datetime.today()

    # Loop through each date and calculate the days elapsed
    for date_str in dividend_dates:
        dividend_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_elapsed = (today - dividend_date).days
        time_elapsed_since_div.append(days_elapsed)

    return time_elapsed_since_div
