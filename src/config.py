# System prompt for the chatbot config.py
import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
SYSTEM_PROMPT = """
You are a helpful customer service assistant for an e-commerce platform. Your role is to assist customers with:

1. Checking order status
2. Explaining return policies
3. Answering common questions about shipping, products, and account management
4. Directing customers to human representatives when necessary

Be concise, friendly, and helpful. If you don't know the answer to a specific question, don't make up information - instead, offer to connect the customer with a human representative.

When discussing return policies, use the following general guidelines:
- Most items can be returned within 30 days
- Electronics have a 14-day return window
- Clothing can be returned within 45 days
- Perishable items must be reported within 3 days of receipt if damaged

For order status inquiries, explain what each status means and provide relevant next steps.

IMPORTANT: When checking order status, ask for either order ID or customer ID. Both are 32-character alphanumeric codes.
"""

# Descriptions of order statuses
ORDER_STATUS_DESCRIPTIONS = {
    "created": "Your order has been created but not yet processed. Payment is being verified.",
    "approved": "Your payment has been approved and your order is being prepared for shipping.",
    "processing": "Your order is currently being processed in our warehouse.",
    "shipped": "Your order has been shipped and is on its way to you.",
    "delivered": "Your order has been delivered to the specified address.",
    "canceled": "Your order has been canceled.",
    "unavailable": "Some items in your order are currently unavailable.",
    "invoiced": "Your order has been invoiced and is being prepared for shipping."
}

# Configuration for offline mode
OFFLINE_MODE_CONFIG = {
    "use_cached_data": True,
    "fallback_responses": {
        "order_not_found": "I'm unable to look up your order at the moment. Please try again later or contact customer service at support@example.com.",
        "general_error": "I'm experiencing some technical difficulties. Please try again later.",
        "connection_error": "I'm currently operating in offline mode with limited functionality. Basic questions can be answered, but specific order lookups are unavailable."
    }
}

# API configuration
API_CONFIG = {
    # OpenAI configuration
    "openai": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.2,
        "max_tokens": 500,
        "timeout_seconds": 15
    },
    # Gemini configuration
    "gemini": {
        "model": "gemini-1.5-flash",
        "temperature": 0.2,
        "max_tokens": 500,
        "timeout_seconds": 15
    }
}

# Retry configuration for API calls
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2,  # seconds
    "backoff_factor": 2  # exponential backoff
}