import pandas as pd
import requests
from bs4 import BeautifulSoup

def scrape_indian_cities():
    # Using a reliable source for Indian cities
    url = "https://en.wikipedia.org/wiki/List_of_cities_in_India_by_population"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Find the main table
    table = soup.find('table', {'class': 'wikitable'})
    
    cities = []
    for row in table.find_all('tr')[1:]:  # Skip header row
        cols = row.find_all('td')
        if len(cols) >= 2:
            city_name = cols[1].text.strip()
            cities.append(city_name)
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({'city': cities})
    df.to_csv('cities_raw.csv', index=False)
    print(f"Scraped {len(cities)} Indian cities")

if __name__ == "__main__":
    scrape_indian_cities() 