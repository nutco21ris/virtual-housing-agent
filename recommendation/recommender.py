from .data_fetcher import fetch_rental_data
from .analyzer import filter_and_score_listings, enhanced_filter_listings
from typing import Dict, List, Any
import pandas as pd

def generate_recommendations(df: pd.DataFrame, criteria: Dict[str, Any], weights: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Generate recommendations based on given criteria and weights.
    
    :param df: DataFrame containing rental listings
    :param criteria: Dictionary of filtering criteria
    :param weights: Dictionary of importance weights for different criteria
    :return: List of recommended listings as dictionaries
    """
    try:
        filtered_listings = filter_and_score_listings(df, criteria, weights)
        
        if not filtered_listings.empty:
            recommended_listings = enhanced_filter_listings(filtered_listings.head(20), 0.5)
            return recommended_listings.to_dict('records')
        else:
            return []
    except Exception as e:
        raise RuntimeError(f"Error in recommendation system: {str(e)}")

def fetch_rental_data_if_needed(current_data: pd.DataFrame) -> pd.DataFrame:
    """
    Fetch rental data if the current data is empty.
    
    :param current_data: Current DataFrame of rental listings
    :return: Updated DataFrame of rental listings
    """
    if current_data.empty:
        return fetch_rental_data()
    return current_data

# Assume these functions are defined elsewhere or import them
# from .listing_processor import filter_and_score_listings, enhanced_filter_listings
# from .data_fetcher import fetch_rental_data