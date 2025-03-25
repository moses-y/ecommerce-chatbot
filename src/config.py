# System prompt for the chatbot config.py
"""
Configuration settings for the E-commerce Support Chatbot.
This module contains all the configuration parameters used throughout the application.
"""
import os
import sys
from typing import Dict, Any

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# ===== Chatbot System Prompt =====
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
# ======= FAQ CONFIG ======
FAQ_CONFIG = {
    "responses": {
        "return_policy": """Our return policy is as follows:

1. Items can be returned within 30 days of delivery for a full refund.
2. Products must be in original packaging and unused condition.
3. For electronics, returns are accepted within 15 days and must include all accessories.
4. Shipping costs for returns are covered by the customer unless the item was defective.
5. Refunds are processed within 5-7 business days after we receive the returned item.

Would you like more information about a specific aspect of our return policy?""",

        "shipping_policy": """Our shipping policy:

1. Standard shipping (5-7 business days): Free for orders over $35, otherwise $4.99
2. Express shipping (2-3 business days): $9.99
3. Next-day delivery (where available): $19.99
4. International shipping available to select countries

Delivery times may vary based on your location and product availability. You can track your shipment using the order ID provided in your confirmation email.

Do you have any other questions about shipping?""",

        "payment_methods": """We accept the following payment methods:

1. Credit cards (Visa, Mastercard, American Express, Discover)
2. Debit cards
3. PayPal
4. Store credit/gift cards
5. Apple Pay and Google Pay (on mobile)

All payment information is securely processed and encrypted. We do not store your full credit card details on our servers.

Is there anything specific about our payment options you'd like to know?""",

        "contact_info": """Our contact information:

Customer Service Hours:
- Monday to Friday: 8:00 AM - 8:00 PM EST
- Saturday: 9:00 AM - 6:00 PM EST
- Sunday: 10:00 AM - 5:00 PM EST

Phone: +1-800-123-4567
Email: support@ecommerce-example.com
Live Chat: Available on our website during business hours

For the fastest response, please have your order number ready when contacting us.

Would you like me to connect you with a customer service representative?""",

        "greeting": """Hello! Welcome to our e-commerce support. How can I help you today? You can ask about order status, return policies, shipping information, or connect with a human representative.""",

        "goodbye": """You're welcome! Thank you for contacting our support. If you have any other questions in the future, don't hesitate to reach out. Have a great day!"""
    },
    
    "intent_patterns": {
        "return_policy": [
            "return policy", "policy on return", "can i return",
            "how to return", "policy for returns", "returned items"
        ],
        "shipping_policy": [
            "shipping policy", "delivery policy", "shipping time",
            "how long shipping", "shipping cost", "how long does shipping",
            "shipping take"
        ],
        "payment_methods": [
            "payment method", "payment option", "how to pay",
            "accept payment", "credit card", "debit card", "paypal"
        ],
        "contact_info": [
            "contact info", "contact information", "phone number",
            "email address", "contact us", "customer service contact"
        ],
        "greeting": [
            "hello", "hi", "hey", "greetings", "good morning", 
            "good afternoon", "good evening"
        ],
        "goodbye": [
            "thank you", "thanks", "bye", "goodbye", "see you", "that's all"
        ]
    }
}
# ===== Order Status Descriptions =====
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

# ===== API Configuration =====

GEMINI_CONFIG = {
    "model": "gemini-1.5-pro",  # or "gemini-1.5-flash" depending on your needs
    "temperature": 0.2,
    "max_output_tokens": 1024,
    "top_p": 0.95,
    "top_k": 40,
    "context_window": 15  # Number of previous messages to consider
}

# ===== Conversation Configuration =====

# Configuration for conversation handling
CONVERSATION_CONFIG = {
    "max_history_length": 10,
    "context_window_size": 5,
    "memory_persistence": True,
    "summarize_threshold": 8  # Number of messages before creating a summary
}

# ===== Vector Database Configuration =====
VECTOR_DB_CONFIG = {
    "db_path": "./data/vector_db",
    "collection_name": "orders",
    "embedding_model": "all-MiniLM-L6-v2",
    "batch_size": 5000,
    "hnsw_params": {
        "M": 16,
        "construction_ef": 100
    }
}

# ===== Offline Mode Configuration =====
OFFLINE_MODE_CONFIG = {
    "use_cached_data": True,
    "fallback_responses": {
        "order_not_found": "I'm unable to look up your order at the moment. Please try again later or contact customer service at support@example.com.",
        "general_error": "I'm experiencing some technical difficulties. Please try again later.",
        "connection_error": "I'm currently operating in offline mode with limited functionality. Basic questions can be answered, but specific order lookups are unavailable."
    }
}

# ===== Retry Configuration =====
RETRY_CONFIG = {
    "max_retries": 3,
    "retry_delay": 2,  # seconds
    "backoff_factor": 2  # exponential backoff
}

# ===== Logging Configuration =====
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": "logs/chatbot.log",
    "max_size": 10 * 1024 * 1024,  # 10 MB
    "backup_count": 5
}

# ===== Data Storage Configuration =====
DATA_CONFIG = {
    "order_data_path": "data/olist_orders_dataset.csv",
    "cache_path": "data/cached_orders.csv",
    "contact_requests_path": "data/contact_requests.csv",
    "feedback_path": "data/feedback/"
}

# ===== UI Configuration =====
UI_CONFIG = {
    "theme": "dark",
    "primary_color": "#14B8A6",  # Teal
    "secondary_color": "#6366F1",  # Indigo
    "font": "Inter, system-ui, sans-serif",
    "chat_height": 500,
    "show_avatars": True,
    "show_timestamps": True
}

# ===== Feature Flags =====
FEATURE_FLAGS = {
    "use_vector_search": True,
    "collect_feedback": True,
    "show_suggestions": True,
    "enable_file_uploads": True,
    "enable_voice_input": False,  # Future feature
    "enable_analytics": True
}

# ===== Human Agent Configuration =====
HUMAN_AGENT_CONFIG = {
    "business_hours": {
        "weekdays": {"start": "08:00", "end": "20:00"},  # EST
        "saturday": {"start": "09:00", "end": "18:00"},  # EST
        "sunday": {"start": "10:00", "end": "17:00"}     # EST
    },
    "expected_response_time": "within 24 hours",
    "contact_fields_required": ["name", "email"],
    "contact_fields_optional": ["phone"]
}

# ===== Load Environment-Specific Configuration =====
def load_environment_config() -> Dict[str, Any]:
    """Load environment-specific configuration."""
    env = os.getenv("ENVIRONMENT", "development")

    if env == "production":
        return {
            "debug": False,
            "use_cache": True,
            "log_level": "INFO",
            "analytics_enabled": True
        }
    elif env == "staging":
        return {
            "debug": True,
            "use_cache": True,
            "log_level": "DEBUG",
            "analytics_enabled": True
        }
    else:  # development
        return {
            "debug": True,
            "use_cache": False,
            "log_level": "DEBUG",
            "analytics_enabled": False
        }

# Load environment-specific config
ENV_CONFIG = load_environment_config()
