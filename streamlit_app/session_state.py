import streamlit as st

def initialize_session_state():
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {'view_type': 'Map'}
    if 'messages' not in st.session_state:
        st.session_state.messages = [("assistant", "Hello! Welcome to the San Francisco housing rental service. How can I assist you today in finding the perfect rental property?")]
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}
