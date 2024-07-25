import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from .location_finder import autocomplete_location, get_location_coordinates, search_place, fetch_reviews, calculate_distance
import logging

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_sentiment_score(text):
    return sia.polarity_scores(text)["compound"]

def analyze_reviews(reviews):
    review_data = {
        "review_text": [review['text'] for review in reviews],
        "sentiment_score": [calculate_sentiment_score(review['text']) for review in reviews]
    }
    reviews_df = pd.DataFrame(review_data)
    avg_sentiment_score = reviews_df['sentiment_score'].mean()
    return avg_sentiment_score

def calculate_listing_score(row, criteria, weights, normalized_price=False):
    score = 0
    score += weights['bedrooms'] * (1 if row['bedrooms'] >= criteria['bedrooms'] else 0)
    score += weights['bathrooms'] * (1 if row['bathrooms'] >= criteria['bathrooms'] else 0)
    if normalized_price:
        score += weights['price'] * (1 - row['normalized_price'])
    else:
        score += weights['price'] * (1 - row['price'])
    score += weights['distance'] * (1 - row['distance'])
    return score



def filter_and_score_listings(df, criteria, weights):
    df['original_price'] = df['price'].copy() 
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df[df['price'] > 0]

    filtered_df = df[
        (df['bedrooms'] >= criteria['bedrooms']) &
        (df['bathrooms'] >= criteria['bathrooms']) &
        (df['price'] >= criteria['min_rent']) &
        (df['price'] <= criteria['max_rent'])
    ].copy(deep=True)

    if filtered_df.empty:
        logger.warning("No listings match the initial criteria")
        return filtered_df

    filtered_df = filtered_df[
        (filtered_df['latitude'].notna()) & 
        (filtered_df['longitude'].notna()) & 
        (filtered_df['formattedAddress'].notna()) & 
        (filtered_df['price'] > 0)
    ]

    try:
        predictions = autocomplete_location(criteria['location'])
        if predictions:
            place_id = predictions[0]['place_id']
            lat, lng = get_location_coordinates(place_id)

            filtered_df['distance'] = filtered_df.apply(lambda row: calculate_distance(lat, lng, row['latitude'], row['longitude']), axis=1)
            filtered_df = filtered_df[filtered_df['distance'] <= criteria['max_distance_km']]

        if not filtered_df.empty:
            scaler = MinMaxScaler()
            normalized_price = scaler.fit_transform(filtered_df[['price']])
            filtered_df['normalized_price'] = normalized_price
        
            filtered_df['score'] = filtered_df.apply(
            lambda row: calculate_listing_score(row, criteria, weights, normalized_price=True), 
            axis=1
        )
        filtered_df = filtered_df.sort_values('score', ascending=False)
        
    except Exception as e:
        logger.error(f"Error in filtering and scoring listings: {e}")
        return pd.DataFrame()

    logger.info(f"Filtered listings count: {len(filtered_df)}")
    return filtered_df[['original_price', 'price', 'normalized_price', 'bedrooms', 'bathrooms', 'latitude', 'longitude', 'formattedAddress', 'distance', 'score']]

# Cache to store place IDs and their sentiment scores
place_cache = {}

def get_reviews_for_listings(filtered_df, max_reviews_per_listing=100):
    result_df = filtered_df.copy()
    result_df['place_id'] = result_df['formattedAddress'].apply(lambda x: place_cache.get(x, search_place(x)))
    result_df['sentiment_score'] = 0.0

    for index, row in result_df.iterrows():
        if row['place_id']:
            if row['place_id'] not in place_cache:
                reviews = fetch_reviews(row['place_id'], max_reviews=max_reviews_per_listing)
                if reviews:
                    avg_sentiment_score = analyze_reviews(reviews)
                    place_cache[row['place_id']] = avg_sentiment_score
                else:
                    place_cache[row['place_id']] = 0.0
            result_df.at[index, 'sentiment_score'] = place_cache[row['place_id']]
    return result_df

def enhanced_filter_listings(filtered_df, sentiment):
    if filtered_df.empty:
        logger.warning("No listings to enhance")
        return filtered_df

    result_df = get_reviews_for_listings(filtered_df)
    result_df['enhanced_score'] = (
        result_df['score'] +
        result_df['sentiment_score'] * sentiment)

    return result_df.sort_values('enhanced_score', ascending=False)