from dotenv import load_dotenv
import os
import time
import requests
import pandas as pd
from typing import List, Dict, Any

# Load environment variables
load_dotenv(dotenv_path='/Users/irisyu/Desktop/Project/virtual-housing-agent/.env')

# Access the variables
DATA_API_KEY = os.getenv('DATA_API_KEY')



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

    return pd.DataFrame(all_listings)