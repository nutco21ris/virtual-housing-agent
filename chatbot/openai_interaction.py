import openai
from openai import OpenAI
from .prompt import SYSTEM_PROMPT, Function_description
from dotenv import load_dotenv
import os
from typing import List, Dict, Any

load_dotenv(dotenv_path='/Users/irisyu/Desktop/Project/virtual-housing-agent/.env')

# Access the variables
OPENAI_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_KEY


def get_rental_agent_response(user_prompt: str, session_messages: List[Dict[str, str]]) -> Dict[str, Any]:
    messages = prepare_messages(session_messages, user_prompt)
    
    try:
        chat_completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=[{"type": "function", "function": Function_description}],
            tool_choice="auto"
        )
        
        return chat_completion.choices[0].message
    except Exception as e:
        raise RuntimeError(f"Error in OpenAI API call: {str(e)}")



def prepare_messages(session_messages, user_prompt):
    """
    Prepare the messages for the ChatCompletion API, including system prompt and user messages.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    for msg in session_messages:
        messages.append({"role": "user" if msg[0] == "user" else "assistant", "content": msg[1]})
    
    messages.append({"role": "user", "content": user_prompt})
    
    return messages
