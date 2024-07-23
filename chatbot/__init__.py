"""
This module provides functionality for interacting with a virtual rental agent powered by OpenAI's GPT model.
"""

from .openai_interaction import get_rental_agent_response
from .prompt import SYSTEM_PROMPT, Function_description

__all__ = ['get_rental_agent_response', 'SYSTEM_PROMPT', 'Function_description']


__version__ = "0.1.0"