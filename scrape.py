import pandas as pd

def scrape_indian_cities():
    # Read from local cities.csv file
    df = pd.read_csv("cities.csv")
    
    # Save to cities_raw.csv for training
    df.to_csv('cities_raw.csv', index=False)
    print(f"Processed {len(df)} Indian cities from cities.csv")

if __name__ == "__main__":
    scrape_indian_cities() 