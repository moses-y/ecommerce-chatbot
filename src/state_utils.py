"""Shared state utilities for the chatbot application."""
from datetime import datetime
from typing import Dict, Any

def reset_state() -> Dict[str, Any]:
    """Initialize or reset the chatbot's state."""
    return {
        "messages": [],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_step": 0,
        "chat_history": [],
        "session_id": datetime.now().strftime("%Y%m%d%H%M%S"),
        "feedback": None,
        "type": "messages"
    }

def update_state_from_result(current_state: Dict[str, Any], new_state: Dict[str, Any]) -> Dict[str, Any]:
    """Update current state with new state values."""
    # List of keys to update
    keys_to_update = [
        "messages",
        "order_lookup_attempted",
        "current_order_id",
        "needs_human_agent",
        "contact_info_collected",
        "customer_name",
        "customer_email",
        "customer_phone",
        "contact_step",
        "chat_history"
    ]
    
    for key in keys_to_update:
        if key in new_state:
            current_state[key] = new_state[key]
    
    return current_state