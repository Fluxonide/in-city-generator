import pandas as pd
import requests
import json

def scrape_indian_cities():
    # Using the GitHub JSON file as source
    url = "https://raw.githubusercontent.com/nshntarora/Indian-Cities-JSON/refs/heads/master/cities-name-list.json"
    response = requests.get(url)
    cities = response.json()
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame({'city': cities})
    df.to_csv('cities_raw.csv', index=False)
    print(f"Downloaded {len(cities)} Indian cities")

if __name__ == "__main__":
    scrape_indian_cities() 