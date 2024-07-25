import googlemaps
import os
from dotenv import load_dotenv
import math
import streamlit as st




# Access the variables
PLACES_API_KEY = st.secrets["general"]['PLACES_API_KEY']
GEOCODING_KEY = st.secrets["general"]['GEOCODING_API_KEY']


gmaps_places = googlemaps.Client(key=PLACES_API_KEY)
gmaps_geocoding = googlemaps.Client(key=GEOCODING_KEY)

# Google Maps API Functions
def autocomplete_location(input_text):
    predictions = gmaps_places.places_autocomplete(input_text, types='geocode')
    return predictions

def get_location_coordinates(place_id):
    result = gmaps_geocoding.place(place_id=place_id)
    location = result['result']['geometry']['location']
    return location['lat'], location['lng']

def search_place(query):
    places_result = gmaps_places.places(query)
    if places_result['results']:
        return places_result['results'][0]['place_id']
    return None

def fetch_reviews(place_id, max_reviews=100):
    reviews = []
    place_details = gmaps_places.place(place_id=place_id)
    if 'reviews' in place_details['result']:
        reviews.extend(place_details['result']['reviews'][:max_reviews])
    return reviews

def calculate_distance(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c