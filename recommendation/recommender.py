from .data_fetcher import fetch_rental_data
from .analyzer import filter_and_score_listings, enhanced_filter_listings
from typing import Dict, List, Any
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def generate_recommendations(df: pd.DataFrame, criteria: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    try:
        working_df = df.copy(deep=True)
        filtered_listings = filter_and_score_listings(working_df, criteria, weights)
        
        if not filtered_listings.empty:
            recommended_listings = enhanced_filter_listings(filtered_listings.head(20), 0.5)
            result = recommended_listings[['formattedAddress', 'original_price', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'enhanced_score']].to_dict('records')
            
            for listing in result:
                listing['price'] = listing['original_price']
                del listing['original_price']
            
            return result
        else:
            return []
    except Exception as e:
        logger.error(f"Error in generate_recommendations: {str(e)}")
        return []

def fetch_rental_data_if_needed(current_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch rental data if the current data is empty.

    :param current_data: Current DataFrame of rental listings
    :return: Updated DataFrame of rental listings
    """
    if current_data.empty:
        return fetch_rental_data()
    return current_data
