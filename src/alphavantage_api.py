from datetime import datetime
import requests

"""
My API Key : F1CSEY3K0H1XDJKO
"""

def get_days_elapsed_since_dividends(symbol, api_key):
    """
    Récupère les jours écoulés depuis les dates des dividendes non nuls pour une entreprise donnée.

    :param symbol: Le symbole boursier de l'entreprise (par exemple, 'MSFT' pour Microsoft).
    :param api_key: La clé API pour accéder à Alpha Vantage.
    :return: Une liste des jours écoulés depuis chaque date de dividende non nulle.
    """
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_WEEKLY_ADJUSTED&symbol={symbol}&apikey={api_key}'
    r = requests.get(url)
    data = r.json()

    # Parcourir les données pour trouver les dates avec des dividendes non nuls
    dates_with_dividends = []

    for date, metrics in data["Weekly Adjusted Time Series"].items():
        dividend_amount = float(metrics["7. dividend amount"])
        if dividend_amount > 0:
            dates_with_dividends.append((date, dividend_amount))

    # Liste des dates de dividendes
    dividend_dates = [dates_with_dividends[i][0] for i in range(len(dates_with_dividends))]
    time_elapsed_since_div = []

    # Obtenir la date d'aujourd'hui
    today = datetime.today()

    # Parcourir chaque date et calculer les jours écoulés
    for date_str in dividend_dates:
        dividend_date = datetime.strptime(date_str, "%Y-%m-%d")
        days_elapsed = (today - dividend_date).days
        time_elapsed_since_div.append(days_elapsed)

    return time_elapsed_since_div