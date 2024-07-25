import streamlit as st
import json
import pandas as pd
from recommendation.recommender import generate_recommendations, fetch_rental_data_if_needed
from .session_state import initialize_session_state
from .ui_components import render_chat_interface, render_recommendations, create_map_with_recommendations, format_price


# Streamlit page configuration
st.set_page_config(
    page_title="SF Virtual Housing Agent",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

def add_custom_css():
    st.markdown(
        """
        <style>
        .chat-message {
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
        }
        .user-message {
            background-color: #d4f1f4;
            text-align: left;
        }
        .assistant-message {
            background-color: #f0f0f0;
            text-align: left;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

add_custom_css()
initialize_session_state()

def run_app():
    st.title("🏠 San Francisco Virtual Housing Agent")

    initialize_session_state()

    render_chat_interface()

    with st.form("user_input_form"):
        st.subheader("Please provide your preferences:")
        
        budget_range = st.slider(
            'Select your budget range for monthly rent ($)', 
            1000, 5000, (2000, 3000), 100
        )
        bedrooms = st.select_slider(
            'Select the number of bedrooms', 
            options=[1, 2, 3]
        )
        max_distance = st.slider(
            'Select the maximum distance from your desired location (km)', 
            1, 20, 10, 1
        )
        location = st.text_input("Enter your desired location in San Francisco")
        move_in_date = st.date_input("Select your preferred move-in date")
        lease_term = st.slider(
            'Select the preferred lease term (months)', 
            1, 24, 12, 1
        )

        view_type = st.radio("Select view type", ("Map", "List"))

        submit_button = st.form_submit_button("Submit")

    if submit_button:
        user_info = {
            'budget': budget_range,
            'bedrooms': bedrooms,
            'distance': max_distance,
            'location': location,
            'move_in_date': move_in_date.strftime("%Y-%m-%d"),
            'lease_term': lease_term,
            'view_type': view_type
        }
        st.session_state.user_info = user_info
        process_user_input()

def process_user_input():
    user_info = st.session_state.user_info
    st.session_state.messages.append(("user", json.dumps(user_info)))

    criteria = {
        "bedrooms": user_info['bedrooms'],
        "bathrooms": 1,
        "min_rent": user_info['budget'][0],
        "max_rent": user_info['budget'][1],
        "location": user_info['location'],
        "move_in_date": user_info['move_in_date'],
        "lease_term": user_info['lease_term'],
        "max_distance_km": user_info['distance']
    }
    weights = {
        "bedrooms": 1,
        "bathrooms": 1,
        "price": 1,
        "distance": 1
    }

    with st.spinner("Generating recommendations..."):
        df = fetch_rental_data_if_needed(st.session_state.get('rental_data', pd.DataFrame()))
        
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        df = df[df['price'] > 0]
        
        st.session_state['rental_data'] = df
        recommendations = generate_recommendations(df, criteria, weights)
    
    st.session_state.messages.append(("assistant", "Here are some recommendations based on your criteria:"))
    render_recommendations(recommendations)