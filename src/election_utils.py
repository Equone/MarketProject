import requests
from bs4 import BeautifulSoup
from datetime import datetime

def get_election_dates():
    url = "https://en.wikipedia.org/wiki/List_of_next_general_elections"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    tables = soup.find_all('table', {'class': 'wikitable'})
    election_data = []

    for table in tables:
        for row in table.find_all('tr')[1:]:  # Ignorer l'en-tête de la table
            cells = row.find_all('td')
            if len(cells) >= 5:
                country = cells[0].get_text(strip=True)
                next_presidential = cells[1].get_text(strip=True)
                next_legislative = cells[2].get_text(strip=True)
                last_presidential = cells[3].get_text(strip=True)
                last_legislative = cells[4].get_text(strip=True)

                next_presidential = next_presidential if any(char.isdigit() for char in next_presidential) else "N/A"
                next_legislative = next_legislative if any(char.isdigit() for char in next_legislative) else "N/A"
                last_presidential = last_presidential if any(char.isdigit() for char in last_presidential) else "N/A"
                last_legislative = last_legislative if any(char.isdigit() for char in last_legislative) else "N/A"
                
                if "2026" in last_presidential or "2026" in last_legislative:
                    last_presidential = "N/A"
                    last_legislative = "N/A"

                election_data.append({
                    'Country': country,
                    'Next Presidential': next_presidential,
                    'Next Legislative': next_legislative,
                    'Last Presidential': last_presidential,
                    'Last Legislative': last_legislative
                })
    
    return election_data

def extract_month(date_str):
    parts = date_str.split()
    if len(parts) >= 2:
        return parts[1]
    elif len(parts) == 1:
        return parts[0]
    return "N/A"

def convert_to_month_tuples(election_data):
    result = []
    for item in election_data:
        country = item['Country']
        next_legislative = extract_month(item['Next Legislative'])
        last_presidential = extract_month(item['Last Presidential'])
        result.append((country, next_legislative, last_presidential))
    return result

def days_since_month(month_str, year_str):
    if month_str == 'N/A' or year_str == 'N/A':
        return None
    
    try:
        month_num = datetime.strptime(month_str, '%b').month
    except ValueError:
        return None
    
    today = datetime.today()
    current_year = today.year

    if month_num > today.month:
        target_date = datetime(current_year - 1, month_num, 1)
    else:
        target_date = datetime(current_year, month_num, 1)

    days_elapsed = (today - target_date).days
    return days_elapsed

def get_days_elapsed_for_country(country_name, month_tuples):
    for country, next_legislative_month, last_presidential_month in month_tuples:
        if country == country_name:
            next_legislative_days = days_since_month(next_legislative_month, '2024')
            last_presidential_days = days_since_month(last_presidential_month, '2024')
            return (next_legislative_days, last_presidential_days)
    return None
