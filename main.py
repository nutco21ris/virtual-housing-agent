from streamlit_app.app import run_app
import streamlit as st

if __name__ == "__main__":
    if 'message' not in st.session_state:
        st.session_state.messages = []
    run_app()