# src/core/config.py
import os
from dotenv import load_dotenv
import logging # Added
from pathlib import Path

load_dotenv()
logger = logging.getLogger(__name__) # Added

# --- Essential Credentials ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    logger.error("FATAL: GOOGLE_API_KEY not found in environment variables.")
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
else:
    logger.info("GOOGLE_API_KEY loaded successfully.")

# --- Database Configuration ---
# --- Database Configuration ---
IS_HUGGINGFACE = os.environ.get('HF_SPACE') == 'true'

if IS_HUGGINGFACE:
    # Use absolute paths for Hugging Face
    DATA_DIR = "/app/data"
else:
    # Use relative paths for local development
    ROOT_DIR = Path(__file__).parent.parent.parent
    DATA_DIR = os.path.join(ROOT_DIR, "data")

DATABASE_NAME = "chatbot_data.db"
DB_PATH = os.path.join(DATA_DIR, DATABASE_NAME)
# SQLAlchemy Database URL
SQLALCHEMY_DATABASE_URL = f"sqlite:///{os.path.abspath(DB_PATH)}" # Use absolute path for SQLAlchemy/Alembic
logger.info(f"Database path configured: {SQLALCHEMY_DATABASE_URL}")

# --- Data File Paths ---
ORDERS_CSV_PATH = os.path.join(DATA_DIR, "cached_orders.csv") # Using your provided CSV
POLICIES_JSON_PATH = os.path.join(DATA_DIR, "policies.json")
CONTACTS_TABLE_NAME = "contact_requests" # Table name for contacts in DB
ORDERS_TABLE_NAME = "orders" # Table name for orders in DB

# --- Logging Configuration (Basic Example) ---
LOGGING_LEVEL = logging.INFO # Change to DEBUG for more detail
logging.basicConfig(level=LOGGING_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)
logger.info(f"Ensured data directory exists: {os.path.abspath(DATA_DIR)}")

# --- Gemini LLM Configuration (Adapted from your GEMINI_CONFIG) ---
GEMINI_MODEL_NAME = "gemini-1.5-pro" # Or "gemini-1.5-flash"
GEMINI_TEMPERATURE = 0.2
GEMINI_MAX_OUTPUT_TOKENS = 1024
GEMINI_TOP_P = 0.95
GEMINI_TOP_K = 40

# --- Conversation Settings (Adapted from CONVERSATION_CONFIG) ---
MAX_CONVERSATION_HISTORY = 10 # How many messages (user + assistant) to keep in memory
# System prompt - adapt as needed
SYSTEM_PROMPT = """
You are Ari, a friendly and efficient customer service assistant for our e-commerce platform.
Your primary goals are:
1.  **Order Status:** Look up order status using the provided 32-character alphanumeric order ID.
2.  **Return Policies:** Explain the return policy based on the provided information.
3.  **Human Representative:** If the user asks to speak to a human, or if you cannot handle their request, politely collect their full name, email, and phone number to create a support ticket.

Guidelines:
- Be concise and clear.
- If asked for order status, *always* ask for the 32-character order ID if not provided.
- If you look up an order, provide the status and key dates (purchase, estimated delivery, actual delivery if applicable).
- Use the provided return policy information accurately. Do not invent policies.
- When collecting contact info, ask for name, then email, then phone number, one step at a time. Confirm once collected.
- If you don't know the answer or cannot perform the task, offer to connect the user with a human representative.
- Maintain a friendly and helpful tone.
"""

# --- Order Status Descriptions (Copied from your config) ---
ORDER_STATUS_DESCRIPTIONS = {
    "created": "Your order has been created but not yet processed. Payment is being verified.",
    "approved": "Your payment has been approved and your order is being prepared for shipping.",
    "processing": "Your order is currently being processed in our warehouse.",
    "shipped": "Your order has been shipped and is on its way to you.",
    "delivered": "Your order has been delivered to the specified address.",
    "canceled": "Your order has been canceled.",
    "unavailable": "Some items in your order are currently unavailable.",
    "invoiced": "Your order has been invoiced and is being prepared for shipping."
    # Add any other statuses present in cached_orders.csv if needed
}

# --- Logging Configuration (Basic) ---
LOGGING_LEVEL = logging.INFO # Change to DEBUG for more detail