import streamlit as st
import openai
import pandas as pd
import numpy as np
import googlemaps
import requests
import time
import json
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.preprocessing import MinMaxScaler
import folium
from folium.plugins import MarkerCluster
import os
from dotenv import load_dotenv
import webbrowser
import math

# Download NLTK data
nltk.download('vader_lexicon', quiet=True)
sia = SentimentIntensityAnalyzer()

# Load environment variables
load_dotenv(dotenv_path='/Users/irisyu/Desktop/Project/virtual-housing-agent/.env')

# Access the variables
DATA_API_KEY = os.getenv('DATA_API_KEY')
PLACES_API_KEY = os.getenv('PLACES_API_KEY')
GEOCODING_KEY = os.getenv('GEOCODING_API_KEY')
OPENAI_KEY = os.getenv('OPENAI_API_KEY')

openai.api_key = OPENAI_KEY

gmaps_places = googlemaps.Client(key=PLACES_API_KEY)
gmaps_geocoding = googlemaps.Client(key=GEOCODING_KEY)

# Streamlit page configuration
st.set_page_config(
    page_title="SF Virtual Housing Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap');
    
    body {
        font-family: 'Source Sans Pro', sans-serif;
        background-color: #f0f2f6;
        color: #1e1e1e;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #1e1e1e;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        padding: 10px 15px;
        font-size: 16px;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border: none;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background-color: #45a049;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
        max-width: 80%;
    }
    .user-message {
        background-color: #DCF8C6;
        align-self: flex-end;
    }
    .bot-message {
        background-color: #FFFFFF;
        align-self: flex-start;
    }
    </style>
    """, unsafe_allow_html=True)

# Data API Functions
@st.cache_data
def fetch_sf_rental_listings(limit=500, max_requests=500):
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

    all_listings = []
    request_count = 0

    try:
        for _ in range(max_requests):
            response = requests.get(Data_URL, headers=headers, params=params)
            request_count += 1
            
            if response.status_code == 200:
                listings = response.json()
                if not isinstance(listings, list):
                    st.warning("Unexpected data format received.")
                    break
                
                all_listings.extend(listings)
                
                if len(listings) < params['limit']:
                    break
                
                params['offset'] += len(listings)
            else:
                st.error(f"Error: {response.status_code}")
                st.error(response.text)
                break
            
            time.sleep(1)

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred: {e}")

    return pd.DataFrame(all_listings)

# Google Maps API Functions
def autocomplete_place(input_text):
    predictions = gmaps_places.places_autocomplete(input_text, types='geocode')
    return predictions

def get_lat_lng_from_place_id(place_id):
    result = gmaps_geocoding.place(place_id=place_id)
    location = result['result']['geometry']['location']
    return location['lat'], location['lng']

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def fetch_reviews(place_id, max_reviews=100):
    reviews = []
    place_details = gmaps_places.place(place_id=place_id)
    if 'reviews' in place_details['result']:
        reviews.extend(place_details['result']['reviews'][:max_reviews])
    return reviews

def sentiment_score(text):
    return sia.polarity_scores(text)["compound"]

def search_place(query):
    places_result = gmaps_places.places(query)
    if places_result['results']:
        return places_result['results'][0]['place_id']
    return None

def analyze_reviews(reviews):
    review_data = {
        "review_text": [review['text'] for review in reviews],
        "sentiment_score": [sentiment_score(review['text']) for review in reviews]
    }
    reviews_df = pd.DataFrame(review_data)
    avg_sentiment_score = reviews_df['sentiment_score'].mean()
    return avg_sentiment_score

# Recommendation System Functions
def calculate_score(row, criteria, weights):
    score = 0
    score += weights['bedrooms'] * (1 if row['bedrooms'] >= criteria['bedrooms'] else 0)
    score += weights['bathrooms'] * (1 if row['bathrooms'] >= criteria['bathrooms'] else 0)
    score += weights['price'] * (1 - row['price'])
    score += weights['distance'] * (1 - row['distance'])
    return score

def filter_listings(df, criteria, weights):
    df['original_price'] = df['price'].copy()
    filtered_df = df[
        (df['bedrooms'] >= criteria['bedrooms']) &
        (df['bathrooms'] >= criteria['bathrooms']) &
        (df['price'] >= criteria['min_rent']) &
        (df['price'] <= criteria['max_rent'])
    ].copy()

    if filtered_df.empty:
        return filtered_df

    try:
        predictions = autocomplete_place(criteria['location'])
        if predictions:
            place_id = predictions[0]['place_id']
            lat, lng = get_lat_lng_from_place_id(place_id)

            filtered_df['distance'] = filtered_df.apply(lambda row: haversine(lat, lng, row['latitude'], row['longitude']), axis=1)
            filtered_df = filtered_df[filtered_df['distance'] <= criteria['max_distance_km']]

        if not filtered_df.empty:
            scaler = MinMaxScaler()
            filtered_df[['price', 'distance']] = scaler.fit_transform(filtered_df[['price', 'distance']])

            filtered_df['score'] = filtered_df.apply(lambda row: calculate_score(row, criteria, weights), axis=1)
            filtered_df = filtered_df.sort_values('score', ascending=False)
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return pd.DataFrame()

    return filtered_df

# Cache to store place IDs and their sentiment scores
place_cache = {}

def get_reviews_for_listings(filtered_df, max_reviews_per_listing=100):
    filtered_df['place_id'] = filtered_df['formattedAddress'].apply(lambda x: place_cache.get(x, search_place(x)))
    filtered_df['sentiment_score'] = 0.0

    for index, row in filtered_df.iterrows():
        if row['place_id']:
            if row['place_id'] not in place_cache:
                reviews = fetch_reviews(row['place_id'], max_reviews=max_reviews_per_listing)
                if reviews:
                    avg_sentiment_score = analyze_reviews(reviews)
                    place_cache[row['place_id']] = avg_sentiment_score
                else:
                    place_cache[row['place_id']] = 0.0
            filtered_df.at[index, 'sentiment_score'] = place_cache[row['place_id']]
    return filtered_df

def enhanced_filter_listings(filtered_df, sentiment):
    if filtered_df.empty:
        return filtered_df

    filtered_df = get_reviews_for_listings(filtered_df)
    filtered_df['enhanced_score'] = (
        filtered_df['score'] +
        filtered_df['sentiment_score'] * sentiment)

    return filtered_df.sort_values('enhanced_score', ascending=False)

def run_recommendation_system(criteria, weights):
    try:
        st.write("Debug - Entering run_recommendation_system")
        st.write(f"Debug - Criteria: {criteria}") 
        st.write(f"Debug - Weights: {weights}")
        
        if 'df' not in st.session_state or st.session_state.df.empty:
            st.warning("No rental listings data available. Fetching data...")
            st.session_state.df = fetch_sf_rental_listings()
        
        filtered_listings = filter_listings(st.session_state.df, criteria, weights)
        st.write(f"Debug - Filtered listings count: {len(filtered_listings)}") 
        
        if not filtered_listings.empty:
            recommended_listings = enhanced_filter_listings(filtered_listings.head(20), 0.5)
            st.write(f"Debug - Recommended listings count: {len(recommended_listings)}") 
            return recommended_listings.to_dict('records')
        else:
            st.warning("No listings found that match your criteria.")
            return []
    except Exception as e:
        st.error(f"Error in recommendation system: {e}")
        st.write(f"Debug - Error details: {str(e)}")  
        return []

# Function to get chat response
def get_chat_response(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a virtual San Francisco housing rental agent. Your role is to assist users in finding the perfect rental property by asking for their requirements, preferences, and priorities. Guide the conversation through several stages: gathering basic requirements, understanding preferences, discussing lifestyle needs. Be friendly, informative, and attentive to the user's needs. Only ask for information that hasn't been provided yet. When you have gathered all necessary information, call the run_recommendation_system function."
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]

    func_descrip = [{
        "name": "run_recommendation_system",
        "description": "Filters and recommends property listings based on user criteria and weights",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "object",
                    "description": "Dictionary of filtering criteria for property listings",
                    "properties": {
                        "bedrooms": {"type": "number", "description": "Number of bedrooms"},
                        "bathrooms": {"type": "number", "description": "Number of bathrooms"},
                        "min_rent": {"type": "number", "description": "Minimum rent price"},
                        "max_rent": {"type": "number", "description": "Maximum rent price"},
                        "location": {"type": "string", "description": "Desired location"},
                        "move_in_date": {"type": "string", "description": "Move-in date (YYYY-MM-DD)"},
                        "lease_term": {"type": "number", "description": "Lease term in months"},
                        "max_distance_km": {"type": "number", "description": "Maximum distance from the location in kilometers"}
                    },
                    "required": ["bedrooms", "bathrooms", "min_rent", "max_rent", "location", "move_in_date", "lease_term", "max_distance_km"]
                },
                "weights": {
                    "type": "object",
                    "description": "Dictionary of importance weights for different criteria",
                    "properties": {
                        "bedrooms": {"type": "number", "description": "Importance of number of bedrooms"},
                        "bathrooms": {"type": "number", "description": "Importance of number of bathrooms"},
                        "price": {"type": "number", "description": "Importance of price"},
                        "distance": {"type": "number", "description": "Importance of distance from desired location"}
                    },
                    "required": ["bedrooms", "bathrooms", "price", "distance"]
                }
            },
            "required": ["criteria", "weights"]
        }
    }]

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=func_descrip,
        function_call="auto"
    )

    return chat_completion.choices[0].message

# Visualization function
def visualize_results_on_map(results):
    map_file = "recommended_listings_map.html"
    
    try:
        df = pd.DataFrame(results) if isinstance(results, list) else results
        
        if df.empty:
            st.warning("No listings to display on the map.")
            sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
        else:
            sf_map = folium.Map(location=[37.7749, -122.4194], zoom_start=12)
            marker_cluster = MarkerCluster().add_to(sf_map)

            for rank, (index, row) in enumerate(df.iterrows(), start=1):
                try:
                    price = row.get('original_price', row['price'])
                    folium.Marker(
                        location=[row['latitude'], row['longitude']],
                        popup=(
                            f"Rank: {rank}<br>"
                            f"Address: {row['formattedAddress']}<br>"
                            f"Price: ${price:.2f}<br>"
                            f"Bedrooms: {row['bedrooms']}<br>"
                            f"Bathrooms: {row['bathrooms']}<br>"
                            f"Score: {row['enhanced_score']:.2f}"
                        ),
                        icon=folium.Icon(color='blue')
                    ).add_to(marker_cluster)
                except Exception as e:
                    st.error(f"Error adding marker for listing {rank}: {e}")

        sf_map.save(map_file)
        return os.path.realpath(map_file)
    except Exception as e:
        print(f"An error occurred while creating the map: {e}")
        import traceback
        traceback.print_exc()
        # Create a simple HTML file with an error message
        with open(map_file, 'w') as f:
            f.write(f"<html><body><h1>Error creating map</h1><p>{str(e)}</p></body></html>")
    
    return os.path.realpath(map_file)

def open_map(map_file):
    if map_file and os.path.exists(map_file):
        webbrowser.open('file://' + os.path.realpath(map_file))
    else:
        print(f"Unable to open map. File not found: {map_file}")

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'current_step' not in st.session_state:
    st.session_state.current_step = 0
if 'user_info' not in st.session_state:
    st.session_state.user_info = {}
if 'df' not in st.session_state:
    st.session_state.df = fetch_sf_rental_listings()


st.title("🏠 San Francisco Virtual Housing Agent")


chat_container = st.container()


col1, col2 = st.columns([3, 1])
with col1:
    user_input = st.text_input("You:", key="user_input")
with col2:
    submit_button = st.button("Send")

if submit_button and user_input:
    st.session_state.messages.append(("user", user_input))
    
    with st.spinner("Thinking..."):
        bot_response = get_chat_response(user_input)
    
    st.write("Debug - Bot response:", bot_response) 

    if 'function_call' in bot_response:
        function_call = bot_response['function_call']
        st.write("Debug - Function call:", function_call) 
        if function_call['name'] == "run_recommendation_system":
            try:
                function_args = json.loads(function_call['arguments'])
                criteria = function_args['criteria']
                weights = function_args['weights']
                
                st.write("Debug - Criteria:", criteria)
                st.write("Debug - Weights:", weights)
                
                with st.spinner("Searching for properties..."):
                    recommended_listings = run_recommendation_system(criteria, weights)
                
                st.write(f"Debug - Recommended listings: {len(recommended_listings)}")
                
                if recommended_listings:
                    bot_response_text = f"I've found {len(recommended_listings)} properties based on your criteria. Here are the top 5:"
                    for i, listing in enumerate(recommended_listings[:5], 1):
                        bot_response_text += f"\n\n{i}. {listing['formattedAddress']}\nPrice: ${listing['price']:.2f}\nBedrooms: {listing['bedrooms']}\nBathrooms: {listing['bathrooms']}\nScore: {listing['enhanced_score']:.2f}"

                    with st.spinner("Creating map..."):
                        map_file = visualize_results_on_map(recommended_listings)
                    st.markdown(f"[View Map of Recommendations]({map_file})")
                else:
                    bot_response_text = "I'm sorry, but I couldn't find any properties matching your criteria. Could you please adjust your requirements?"
            except Exception as e:
                st.error(f"An error occurred while processing recommendations: {e}")
                st.write(f"Debug - Error details: {str(e)}")
                bot_response_text = "I apologize, but I encountered an error while trying to find recommendations. Could you please try again or rephrase your request?"
        else:
            bot_response_text = "I'm sorry, I'm not sure how to handle that request."
    else:
        bot_response_text = bot_response['content']
    
    st.session_state.messages.append(("assistant", bot_response_text))
    
    st.experimental_rerun()

with chat_container:
    for role, message in st.session_state.messages:
        if role == "user":
            st.markdown(f'<div class="chat-message user-message">{message}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-message bot-message">{message}</div>', unsafe_allow_html=True)


if not st.session_state.messages:
    welcome_message = "Welcome to the San Francisco Virtual Housing Agent! I'm here to help you find the perfect rental property in San Francisco. Let's start by getting to know your needs and preferences. What brings you to San Francisco?"
    st.session_state.messages.append(("assistant", welcome_message))
    st.experimental_rerun()


def get_chat_response(user_prompt):
    messages = [
        {
            "role": "system",
            "content": "You are a virtual San Francisco housing rental agent. Your role is to assist users in finding the perfect rental property by asking for their requirements, preferences, and priorities. Guide the conversation through several stages: gathering basic requirements, understanding preferences, discussing lifestyle needs. Be friendly, informative, and attentive to the user's needs. Only ask for information that hasn't been provided yet. When you have gathered all necessary information (bedrooms, bathrooms, min_rent, max_rent, location, move_in_date, lease_term, and max_distance_km), call the run_recommendation_system function with appropriate criteria and weights."
        }
    ]
    
    for msg in st.session_state.messages:
        messages.append({"role": "user" if msg[0] == "user" else "assistant", "content": msg[1]})
    
    messages.append({"role": "user", "content": user_prompt})

    func_descrip = [{
        "name": "run_recommendation_system",
        "description": "Filters and recommends property listings based on user criteria and weights",
        "parameters": {
            "type": "object",
            "properties": {
                "criteria": {
                    "type": "object",
                    "description": "Dictionary of filtering criteria for property listings",
                    "properties": {
                        "bedrooms": {"type": "number", "description": "Number of bedrooms"},
                        "bathrooms": {"type": "number", "description": "Number of bathrooms"},
                        "min_rent": {"type": "number", "description": "Minimum rent price"},
                        "max_rent": {"type": "number", "description": "Maximum rent price"},
                        "location": {"type": "string", "description": "Desired location"},
                        "move_in_date": {"type": "string", "description": "Move-in date (YYYY-MM-DD)"},
                        "lease_term": {"type": "number", "description": "Lease term in months"},
                        "max_distance_km": {"type": "number", "description": "Maximum distance from the location in kilometers"}
                    },
                    "required": ["bedrooms", "bathrooms", "min_rent", "max_rent", "location", "move_in_date", "lease_term", "max_distance_km"]
                },
                "weights": {
                    "type": "object",
                    "description": "Dictionary of importance weights for different criteria",
                    "properties": {
                        "bedrooms": {"type": "number", "description": "Importance of number of bedrooms"},
                        "bathrooms": {"type": "number", "description": "Importance of number of bathrooms"},
                        "price": {"type": "number", "description": "Importance of price"},
                        "distance": {"type": "number", "description": "Importance of distance from desired location"}
                    },
                    "required": ["bedrooms", "bathrooms", "price", "distance"]
                }
            },
            "required": ["criteria", "weights"]
        }
    }]

    st.write("Debug - Messages sent to AI:", messages)  

    chat_completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        functions=func_descrip,
        function_call="auto"
    )

    return chat_completion.choices[0].message

if submit_button and user_input:
    st.session_state.messages.append(("user", user_input))
    
    with st.spinner("Thinking..."):
        bot_response = get_chat_response(user_input)
    
    if bot_response.get('function_call'):
        function_call = bot_response['function_call']
        if function_call['name'] == "run_recommendation_system":
            try:
                function_args = json.loads(function_call['arguments'])
                criteria = function_args['criteria']
                weights = function_args['weights']
                
                with st.spinner("Searching for properties..."):
                    recommended_listings = run_recommendation_system(criteria, weights)
                
                if recommended_listings:
                    bot_response_text = f"I've found {len(recommended_listings)} properties based on your criteria. Here are the top 5:"
                    for i, listing in enumerate(recommended_listings[:5], 1):
                        bot_response_text += f"\n\n{i}. {listing['formattedAddress']}\nPrice: ${listing['price']:.2f}\nBedrooms: {listing['bedrooms']}\nBathrooms: {listing['bathrooms']}\nScore: {listing['enhanced_score']:.2f}"

                    with st.spinner("Creating map..."):
                        map_file = visualize_results_on_map(recommended_listings)
                    st.markdown(f"[View Map of Recommendations]({map_file})")
                else:
                    bot_response_text = "I'm sorry, but I couldn't find any properties matching your criteria. Could you please adjust your requirements?"
            except Exception as e:
                st.error(f"An error occurred while processing recommendations: {e}")
                bot_response_text = "I apologize, but I encountered an error while trying to find recommendations. Could you please try again or rephrase your request?"
        else:
            bot_response_text = "I'm sorry, I'm not sure how to handle that request."
    else:
        bot_response_text = bot_response['content']
    
    st.session_state.messages.append(("assistant", bot_response_text))
    
    st.session_state.user_input = ""
    st.experimental_rerun()