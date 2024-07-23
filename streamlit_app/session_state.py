import streamlit as st

def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 0
    if 'user_info' not in st.session_state:
        st.session_state.user_info = {}