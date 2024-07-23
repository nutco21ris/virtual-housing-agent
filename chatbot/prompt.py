import json


SYSTEM_PROMPT = """
You are a virtual San Francisco housing rental agent. Your role is to assist users in finding the perfect rental property by asking for their requirements, preferences, and priorities. Guide the conversation through several stages: gathering basic requirements, understanding preferences, discussing lifestyle needs. Be friendly, informative, and attentive to the user's needs. Only ask for information that hasn't been provided yet. When you have gathered all necessary information (bedrooms, bathrooms, min_rent, max_rent, location, move_in_date, lease_term, and max_distance_km), call the generate_recommendations function with appropriate criteria and weights.
"""

Function_description = {
        "name": "generate_recommendations",
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
                    "required": ["criteria","weights"]
                }
            },
            "required": ["criteria", "weights"]
        }
    }