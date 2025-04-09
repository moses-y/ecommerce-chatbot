# src/services/policy_service.py
import json
import logging
import os
from typing import Dict, Any, Optional
from src.core.config import POLICIES_JSON_PATH

logger = logging.getLogger(__name__)

class PolicyService:
    """Handles loading and retrieving policy information using a class-based cache."""

    _instance = None
    _policies_cache: Optional[Dict[str, Any]] = None

    # --- Singleton Pattern (Optional but ensures only one instance loads the file) ---
    # If you prefer not to use a singleton, remove __new__ and instantiate
    # PolicyService() where needed (e.g., once in your main app setup).
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(PolicyService, cls).__new__(cls)
            # Load policies only when the first instance is created
            cls._instance._load_policies()
        return cls._instance
    # --- End Singleton Pattern ---

    # If not using Singleton, uncomment the __init__ and remove __new__
    # def __init__(self):
    #     """Initializes the PolicyService."""
    #     if PolicyService._policies_cache is None:
    #          self._load_policies()

    def _load_policies(self):
        """Loads policies from the JSON file into the class cache."""
        logger.info(f"Attempting to load policies from: {POLICIES_JSON_PATH}")
        if not os.path.exists(POLICIES_JSON_PATH):
            logger.error(f"Policy file not found at {POLICIES_JSON_PATH}. Policies unavailable.")
            PolicyService._policies_cache = {"error": "Policy information is currently unavailable."}
            return

        try:
            with open(POLICIES_JSON_PATH, 'r', encoding='utf-8') as f:
                policies = json.load(f)
            PolicyService._policies_cache = policies # Cache the loaded policies
            logger.info("Policies loaded successfully into cache.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from {POLICIES_JSON_PATH}: {e}", exc_info=True)
            PolicyService._policies_cache = {"error": "Error reading policy information."}
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading policies: {e}", exc_info=True)
            PolicyService._policies_cache = {"error": "An unexpected error occurred."}

    def get_policy(self, policy_name: str) -> Optional[Dict[str, Any]]:
        """Retrieves a specific policy section from the cache."""
        if PolicyService._policies_cache is None:
            logger.warning("Policy cache is not initialized.")
            # Attempt to load again? Or rely on initial load.
            # self._load_policies() # Uncomment if you want to retry loading on demand
            return None # Or return the error dict if present
        if "error" in PolicyService._policies_cache:
             return None # Policies failed to load initially

        policy_section = PolicyService._policies_cache.get(policy_name)
        if policy_section is None:
             logger.warning(f"Policy section '{policy_name}' not found in cache.")
        return policy_section

    def get_all_policies(self) -> Dict[str, Any]:
        """Returns all loaded policies from the cache."""
        if PolicyService._policies_cache is None:
             logger.warning("Policy cache is not initialized.")
             return {"error": "Policies not loaded."}
        return PolicyService._policies_cache

    def get_policy_summary(self, policy_name: str = "general_return_policy") -> str:
        """
        Provides a basic text summary of a specific policy from the cache.
        Adapts to the structure in the user's policies.json.
        """
        policy = self.get_policy(policy_name) # Use the class method to get from cache

        if policy is None:
            # Check if the cache itself has an error message
            if PolicyService._policies_cache and "error" in PolicyService._policies_cache:
                 return PolicyService._policies_cache["error"]
            logger.warning(f"Could not retrieve policy section '{policy_name}' for summary.")
            return f"Sorry, I couldn't find the details for the '{policy_name}' policy."
        if not isinstance(policy, dict): # Handle case where policy might not be a dict
             logger.warning(f"Policy section '{policy_name}' is not a dictionary: {policy}")
             return f"Sorry, the details for '{policy_name}' policy are not formatted correctly."


        summary_parts = []
        # Use 'window' key from user's JSON
        if 'window' in policy:
            summary_parts.append(f"Return Window: {policy['window']}.")
        if 'condition' in policy:
            summary_parts.append(f"Condition: {policy['condition']}")
        if 'refund_type' in policy:
            summary_parts.append(f"Refunds: {policy['refund_type']}")
        if 'process' in policy:
             summary_parts.append(f"Process: {policy['process']}")
        # Handle 'exceptions' which might be a string or a list
        if 'exceptions' in policy and policy['exceptions']:
            exceptions_text = policy['exceptions']
            if isinstance(exceptions_text, list):
                summary_parts.append("Exceptions include: " + ", ".join(exceptions_text))
            elif isinstance(exceptions_text, str):
                 summary_parts.append(f"Exceptions: {exceptions_text}")
            else:
                 logger.warning(f"Unexpected type for 'exceptions' in policy '{policy_name}': {type(exceptions_text)}")
                 summary_parts.append(f"Exceptions: {str(exceptions_text)}")

        return "\n".join(summary_parts) if summary_parts else f"No summary details available for the '{policy_name}' policy."

    def get_formatted_policies(self) -> str:
        """Returns all loaded policies formatted as a string."""
        policies = self.get_all_policies() # Use class method
        if "error" in policies:
            return policies["error"]

        formatted_output = "Here are our policies:\n"
        for key, value in policies.items():
            # Format key nicely (e.g., general_return_policy -> General Return Policy)
            formatted_key = key.replace('_', ' ').title()
            formatted_output += f"\n### {formatted_key} ###\n"
            if isinstance(value, dict):
                # Format dictionary items
                formatted_output += "\n".join([f"- {k.replace('_', ' ').capitalize()}: {v}" for k, v in value.items()]) + "\n"
            else:
                # Handle non-dict policy sections if they exist
                formatted_output += f"{str(value)}\n"
        return formatted_output.strip() # Remove trailing newline

# Example of how to use (instantiate once in your app)
# policy_service = PolicyService()
# summary = policy_service.get_policy_summary()
# print(summary)
# all_formatted = policy_service.get_formatted_policies()
# print(all_formatted)