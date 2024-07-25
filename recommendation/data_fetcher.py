import os
import time
import requests
import pandas as pd
from typing import List, Dict, Any
import streamlit as st

# Access the variables
DATA_API_KEY = st.secrets["general"]['DATA_API_KEY']

def fetch_rental_data(limit: int = 500, max_requests: int = 500) -> pd.DataFrame:
    Data_URL = 'https://api.rentcast.io/v1/listings/rental/long-term'
    params = {
        'city': 'San Francisco',
        'state': 'CA',
        'limit': limit,
        'status': 'Active', 
        'offset': 0
    }

    headers = {
        'Accept': 'application/json',
        'X-Api-Key': DATA_API_KEY
    }

    all_listings: List[Dict[str, Any]] = []
    request_count = 0

    try:
        for _ in range(max_requests):
            response = requests.get(Data_URL, headers=headers, params=params)
            request_count += 1

            if response.status_code == 200:
                listings = response.json()
                if not isinstance(listings, list):
                    raise ValueError("Unexpected data format received.")

                all_listings.extend(listings)

                if len(listings) < params['limit']:
                    break

                params['offset'] += len(listings)
            else:
                raise requests.HTTPError(f"HTTP error {response.status_code}: {response.text}")

            time.sleep(1)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"An error occurred while fetching rental data: {e}")

    if not all_listings:
        return pd.DataFrame()  # Return an empty DataFrame if no listings are fetched

    df = pd.DataFrame(all_listings)
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    required_columns = ['bedrooms', 'bathrooms', 'price', 'latitude', 'longitude', 'formattedAddress']
    for column in required_columns:
        if column not in df.columns:
            df[column] = None  # or set a default value if appropriate
    numeric_columns = ['bedrooms', 'bathrooms', 'price', 'latitude', 'longitude']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
