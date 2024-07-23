import streamlit as st
from .session_state import initialize_session_state
from .ui_components import render_chat_interface, render_recommendations
from chatbot.openai_interaction import get_rental_agent_response

# Streamlit page configuration
def run_app():
    st.set_page_config(
        page_title="SF Virtual Housing Agent",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
)

    initialize_session_state()

    st.title("🏠 San Francisco Virtual Housing Agent")

    user_input = render_chat_interface()

    if user_input:
        process_user_input(user_input)

def process_user_input(user_input):
    st.session_state.messages.append(("user", user_input))
    
    with st.spinner("Thinking..."):
        bot_response = get_rental_agent_response(user_input, st.session_state.messages)
    
    handle_bot_response(bot_response)

def handle_bot_response(bot_response):
    if 'function_call' in bot_response:
        handle_function_call(bot_response['function_call'])
    else:
        st.session_state.messages.append(("assistant", bot_response['content']))
    
    st.experimental_rerun()

def handle_function_call(function_call):
    if function_call['name'] == "generate_recommendations":
        generate_and_display_recommendations(function_call['arguments'])
    else:
        st.session_state.messages.append(("assistant", "I'm sorry, I'm not sure how to handle that request."))

def generate_and_display_recommendations(arguments):
    pass

if __name__ == "__main__":
    run_app()