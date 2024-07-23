import streamlit as st

def render_chat_interface():
    chat_container = st.container()

    with chat_container:
        for role, message in st.session_state.messages:
            st.markdown(f'<div class="chat-message {role}-message">{message}</div>', unsafe_allow_html=True)

    user_input = st.text_input("You:", key="user_input")
    submit_button = st.button("Send")

    return user_input if submit_button and user_input else None

def render_recommendations(recommended_listings):
    pass