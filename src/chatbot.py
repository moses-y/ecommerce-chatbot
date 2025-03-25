# src/chatbot.py
"""
E-commerce support chatbot using a state graph for conversation management.
Implements a hexagonal architecture with clear separation of concerns.
"""
import os
import sys
import time
from datetime import datetime
import re
import pandas as pd
import logging
import warnings
from functools import lru_cache
from typing import Dict, List, Any, Optional, TypedDict, Callable, Tuple

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

# Third-party imports
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Load environment variables
load_dotenv()

# Import local modules
from src.config import ORDER_STATUS_DESCRIPTIONS,CONVERSATION_CONFIG, FAQ_CONFIG
from src.utils import load_order_data, get_order_status, format_order_details, create_order_index
from src.state_management import ConversationMemory
from src.llm_service import LLMService
from src.state_utils import reset_state, update_state_from_result

# Import vector database functions - these are imported here to avoid circular imports
# The vector_db module should not import from chatbot
from src.vector_db import (
    get_vector_db_instance,
    query_vector_db,
    get_order_by_id,
    get_orders_by_customer_id,
    cached_get_order_by_id,
    cached_get_orders_by_customer_id
)

conversation_memory = ConversationMemory(
    max_history=CONVERSATION_CONFIG["max_history_length"]
)

# Add this after the imports to maintain backward compatibility
FAQ_RESPONSES = FAQ_CONFIG["responses"]
# ===== Cached Data =====

# Load cached data
@lru_cache(maxsize=1)
def load_order_data_cached():
    """Load order data with caching for better performance."""
    return load_order_data()

# ===== Domain Models =====

class ChatbotState(TypedDict):
    """Type definition for the chatbot state."""
    messages: List[Dict[str, Any]]
    order_lookup_attempted: bool
    current_order_id: Optional[str]
    needs_human_agent: bool
    contact_info_collected: bool
    customer_name: Optional[str]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    contact_step: int
    chat_history: List[Dict[str, Any]]
    session_id: str
    feedback: Optional[str]
    type: str

class OrderService:
    """Service for handling order-related operations."""

    def __init__(self, orders_df=None, vector_collection=None):
        """Initialize the order service."""
        # Load order data if not provided
        if orders_df is None:
            self.orders_df = load_order_data_cached()
        else:
            self.orders_df = orders_df

        # Create indexes for faster lookups
        self.order_index, self.customer_index = create_order_index(self.orders_df)

        # Get vector collection if not provided
        if vector_collection is None:
            self.vector_collection = get_vector_db_instance(self.orders_df)
        else:
            self.vector_collection = vector_collection

    def lookup_order_by_id(self, order_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Look up an order by ID."""
        # Try direct dictionary lookup first (fastest)
        if order_id in self.order_index:
            order_data = self.order_index[order_id]
            status = order_data['order_status']
            details = {
                'purchase_date': order_data['order_purchase_timestamp'],
                'delivery_date': order_data['order_delivered_customer_date'],
                'approved_date': order_data['order_approved_at'],
                'estimated_delivery': order_data['order_estimated_delivery_date'],
                'actual_delivery': order_data['order_delivered_customer_date']
            }
            return status, details

        # Fall back to vector search
        order_result = cached_get_order_by_id(order_id, self.vector_collection)
        if order_result:
            metadata = order_result["metadata"]
            status = metadata["status"]
            details = {
                'purchase_date': metadata['purchase_date'],
                'delivery_date': metadata.get('customer_delivery_date'),
                'approved_date': metadata.get('approved_date'),
                'estimated_delivery': metadata.get('estimated_delivery_date'),
                'actual_delivery': metadata.get('customer_delivery_date')
            }
            return status, details

        return None, None

    def lookup_orders_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        """Look up orders by customer ID."""
        # Try direct dictionary lookup first (fastest)
        if customer_id in self.customer_index:
            return self.customer_index[customer_id]

        # Fall back to vector search
        return cached_get_orders_by_customer_id(customer_id, self.vector_collection)

class ContactService:
    """Service for handling contact-related operations."""

    def save_contact_info(self, name: str, email: str, phone: str) -> bool:
        """Save contact information to CSV."""
        try:
            import csv
            os.makedirs("data", exist_ok=True)
            csv_path = os.path.join("data", "contact_requests.csv")
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['name', 'email', 'phone']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    'name': name,
                    'email': email,
                    'phone': phone
                })
            return True
        except Exception as e:
            logger.error(f"Error saving contact info: {e}")
            return False

# ===== Initialize Services =====
# ===== Service Layer =====

# Initialize services (do this only once)
llm_service = LLMService()
order_service = OrderService()
contact_service = ContactService()


# ===== Node Functions =====

def lookup_order(state: ChatbotState) -> ChatbotState:
    """Extract order ID or customer ID and look up status."""
    start_time = time.time()
    last_message = state["messages"][-1]["content"]
    new_state = state.copy()

    # Simple regex pattern to extract order ID or customer ID
    id_match = re.search(r'\b([a-f0-9]{32})\b', last_message)

    if id_match:
        extracted_id = id_match.group(1)

        # Try as order ID first
        status, details = order_service.lookup_order_by_id(extracted_id)

        if status:
            # Found as order ID
            response = format_order_details(extracted_id, status, details)
        else:
            # Try as customer ID
            customer_orders = order_service.lookup_orders_by_customer_id(extracted_id)

            if customer_orders:
                if len(customer_orders) == 1:
                    # Only one order for this customer
                    order_data = customer_orders[0]
                    order_id = order_data['order_id']
                    status = order_data['order_status']
                    details = {
                        'purchase_date': order_data['order_purchase_timestamp'],
                        'delivery_date': order_data['order_delivered_customer_date'],
                        'approved_date': order_data['order_approved_at'],
                        'estimated_delivery': order_data['order_estimated_delivery_date'],
                        'actual_delivery': order_data['order_delivered_customer_date']
                    }
                    response = f"I found an order for your customer ID. {format_order_details(order_id, status, details)}"
                else:
                    # Multiple orders for this customer
                    sorted_orders = sorted(customer_orders,
                                          key=lambda x: pd.to_datetime(x['order_purchase_timestamp']),
                                          reverse=True)
                    recent_order = sorted_orders[0]
                    order_id = recent_order['order_id']
                    status = recent_order['order_status']
                    details = {
                        'purchase_date': recent_order['order_purchase_timestamp'],
                        'delivery_date': recent_order['order_delivered_customer_date'],
                        'approved_date': recent_order['order_approved_at'],
                        'estimated_delivery': recent_order['order_estimated_delivery_date'],
                        'actual_delivery': recent_order['order_delivered_customer_date']
                    }
                    response = f"I found {len(customer_orders)} orders for your customer ID. Here's the status of your most recent order: {format_order_details(order_id, status, details)}"
            else:
                # Not found as either order ID or customer ID
                response = f"I couldn't find any orders with ID {extracted_id}. Please check the ID and try again."
    else:
        # No ID found in message
        response = "I'd be happy to help check your order status. Could you please provide your order ID or customer ID? It should be a 32-character alphanumeric code."

    logger.info(f"Lookup completed in {time.time() - start_time:.4f} seconds")

    new_state["messages"].append({"role": "assistant", "content": response})
    new_state["order_lookup_attempted"] = True
    new_state["current_order_id"] = extracted_id if 'extracted_id' in locals() and extracted_id != "NO_ID" else None
    return new_state

def collect_contact_info(state: ChatbotState) -> ChatbotState:
    """Collect customer contact information for human handoff and save to CSV."""
    new_state = state.copy()
    last_message = state["messages"][-1]["content"]
    last_message_lower = last_message.lower()

    # Check if user is trying to do something else instead
    cancel_keywords = ["cancel", "nevermind", "never mind", "stop", "go back"]
    customer_id_keywords = ["customer id", "my id", "delivery status", "order status", "track"]

    if any(keyword in last_message_lower for keyword in cancel_keywords) or any(keyword in last_message_lower for keyword in customer_id_keywords):
        # Reset the human agent flow
        new_state["needs_human_agent"] = False
        new_state["contact_step"] = 0

        # If they're asking about customer ID specifically
        if any(keyword in last_message_lower for keyword in customer_id_keywords):
            new_state["messages"].append({
                "role": "assistant",
                "content": "I understand you'd like to check your order status using your customer ID. Could you please provide your customer ID? It should be a 32-character alphanumeric code."
            })
            return new_state
        else:
            new_state["messages"].append({
                "role": "assistant",
                "content": "I've canceled the request to speak with a human representative. How else can I help you today?"
            })
            return new_state

    # Initialize contact_step if not present
    if "contact_step" not in new_state:
        new_state["contact_step"] = 0

    # Get current step
    step = new_state["contact_step"]

    if step == 0:
        # First time - ask for name
        new_state["contact_step"] = 1
        new_state["messages"].append({
            "role": "assistant",
            "content": "I'll connect you with a human representative. Could you please provide your name?"
        })
    elif step == 1:
        # Got name, ask for email
        new_state["customer_name"] = last_message.strip()
        new_state["contact_step"] = 2
        new_state["messages"].append({
            "role": "assistant",
            "content": f"Thank you, {new_state['customer_name']}. Could you please provide your email address?"
        })
    elif step == 2:
        # Got email, ask for phone
        new_state["customer_email"] = last_message.strip()
        new_state["contact_step"] = 3
        new_state["messages"].append({
            "role": "assistant",
            "content": "Thank you. Finally, could you please provide your phone number?"
        })
    elif step == 3:
        # Got phone, complete collection
        new_state["customer_phone"] = last_message.strip()
        new_state["contact_info_collected"] = True
        new_state["contact_step"] = 4

        # Save contact information to CSV
        success = contact_service.save_contact_info(
            new_state["customer_name"],
            new_state["customer_email"],
            new_state["customer_phone"]
        )

        if success:
            new_state["messages"].append({
                "role": "assistant",
                "content": f"Thank you for providing your information. A customer service representative will contact you soon at {new_state['customer_email']} or {new_state['customer_phone']}. Is there anything else you'd like to add before I submit your request?"
            })
        else:
            new_state["messages"].append({
                "role": "assistant",
                "content": "Thank you for providing your information. I encountered an issue saving your contact details, but a customer service representative will be notified. Is there anything specific you'd like me to tell them about your inquiry?"
            })

    return new_state

def continue_conversation(state: ChatbotState) -> ChatbotState:
    """Process the user message and generate a response."""
    messages = state["messages"]
    last_message = messages[-1]["content"].lower()
    new_state = state.copy()

    # Check if user wants to speak to a human
    human_keywords = ["speak to a human", "talk to a human", "human representative",
                     "real person", "speak to an agent", "talk to a representative",
                     "connect me with a human", "human agent please", "agent", "representative"]

    # More precise check for human requests - avoid triggering on return policy questions
    needs_human = False
    for phrase in human_keywords:
        if phrase in last_message:
            needs_human = True
            break

    # Also check for standalone keywords, but be more careful
    if not needs_human:
        standalone_words = ["human", "representative", "agent", "person"]
        # Only trigger if these words appear alone or as main request
        # Avoid triggering when asking about policies related to these words
        if any(word in last_message.split() for word in standalone_words) and not any(policy_word in last_message for policy_word in ["policy", "policies", "return", "shipping", "warranty"]):
            needs_human = True

    if needs_human:
        new_state["needs_human_agent"] = True
        # Initialize contact step
        new_state["contact_step"] = 0
        return new_state

    # Generate response using LLM service
    response_text = llm_service.generate_response(messages)

    # Update state
    new_state["messages"].append({"role": "assistant", "content": response_text})
    return new_state

def connect_to_agent(state: ChatbotState) -> ChatbotState:
    """Immediately connect to a human agent without collecting contact info."""
    new_state = state.copy()

    # Add a direct connection message
    new_state["messages"].append({
        "role": "assistant",
        "content": "I'm connecting you to a human representative now. Please hold while I transfer your chat. A customer service agent will be with you shortly."
    })

    # Mark as needing human agent but skip contact collection
    new_state["needs_human_agent"] = True
    new_state["contact_info_collected"] = True  # Skip collection

    return new_state

# ===== Router Function =====

def router(state: ChatbotState):
    """Central routing function to determine next node."""
    # Check for human agent requests first - this should take priority
    last_message = state["messages"][-1]["content"].lower()
    human_keywords = ["speak to a human", "talk to a human", "human representative",
                     "real person", "speak to an agent", "talk to a representative",
                     "connect me with a human", "human agent please", "agent", "representative"]

    if any(keyword in last_message for keyword in human_keywords) and not state["needs_human_agent"]:
        # Set flag for human agent
        state["needs_human_agent"] = True
        state["contact_step"] = 0
        return "collect_contact_info"

    # Check if we need to collect contact info for human handoff
    if state["needs_human_agent"] and not state["contact_info_collected"]:
        return "collect_contact_info"

    # Check if we should end the conversation
    if "bye" in last_message or "thank you" in last_message or "thanks" in last_message:
        return END

    # End if contact info is collected
    if state["needs_human_agent"] and state["contact_info_collected"]:
        return END

    # Check if we need to look up an order - prioritize ID detection
    if not state["order_lookup_attempted"]:
        # Check for specific keywords related to order status
        order_related = "order" in last_message and ("status" in last_message or "where" in last_message or "track" in last_message)

        # Also check for direct order ID in the message
        has_id = re.search(r'\b([a-f0-9]{32})\b', last_message) is not None

        if order_related or has_id:
            return "lookup_order"

    # Default to continuing the conversation
    return "continue_conversation"

# ===== Chatbot Creation =====

def create_chatbot():
    """Initialize and return the e-commerce support chatbot with enhanced conversation handling."""
    workflow = StateGraph(ChatbotState)
    
    def enhanced_continue_conversation(state: ChatbotState) -> ChatbotState:
        """Enhanced conversation handling with better context management."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        new_state = state.copy()
        
        # Update conversation memory
        conversation_memory.add_message("user", last_message)
        
        # Update key details if available
        if state.get("current_order_id"):
            conversation_memory.update_key_details("current_order_id", state["current_order_id"])
        if state.get("customer_name"):
            conversation_memory.update_key_details("customer_name", state["customer_name"])
        
        # Generate response using enhanced LLM service
        response_text = llm_service.generate_response(messages, conversation_memory)
        
        # Update conversation memory with assistant's response
        conversation_memory.add_message("assistant", response_text)
        
        # Update state
        new_state["messages"].append({"role": "assistant", "content": response_text})
        return new_state
    
    # Update the workflow with the enhanced conversation handler
    workflow.add_node("continue_conversation", enhanced_continue_conversation)
    workflow.add_node("lookup_order", lookup_order)
    workflow.add_node("collect_contact_info", collect_contact_info)
    
    # Add conditional edges
    workflow.add_conditional_edges("continue_conversation", router)
    workflow.add_conditional_edges("lookup_order", router)
    workflow.add_conditional_edges("collect_contact_info", router)
    
    workflow.set_entry_point("continue_conversation")
    
    return workflow.compile()

# ===== Intent Detection =====

def detect_intent(message: str) -> Optional[str]:
    """Detect the intent of a user message."""
    message_lower = message.lower()
    
    # Check each intent's patterns first
    for intent, patterns in FAQ_CONFIG["intent_patterns"].items():
        if any(pattern in message_lower for pattern in patterns):
            return intent
    
    # Check for human agent request
    human_keywords = [
        "speak to a human", "talk to a human", "human representative",
        "real person", "speak to an agent", "talk to a representative",
        "connect me with a human", "human agent please", "agent", "representative"
    ]
    if any(keyword in message_lower for keyword in human_keywords):
        return "human_agent"

    # Check for order status
    if "order status" in message_lower or (
        "order" in message_lower and "status" in message_lower
    ):
        return "order_status"

    return None


# ===== Main Chat Function =====

def chat_with_user(user_input: str, chat_state: Optional[Dict] = None) -> Dict:
    """Process a single user message and return updated state."""
    if chat_state is None:
        chat_state = reset_state()

    # Only add user message if not already present
    if not chat_state["messages"] or chat_state["messages"][-1]["content"] != user_input:
        chat_state["messages"].append({"role": "user", "content": user_input})

    # Check for FAQs FIRST, before any other processing
    intent = detect_intent(user_input.lower())  # Convert to lowercase once
    if intent and intent in FAQ_RESPONSES and intent not in ["greeting", "goodbye", "human_agent", "order_status"]:
        # Only process direct FAQ responses here
        chat_state["messages"].append({"role": "assistant", "content": FAQ_RESPONSES[intent]})
        return chat_state

    # Check if user wants to cancel human agent request
    if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
        cancel_keywords = ["cancel", "nevermind", "never mind", "stop", "go back", "different question"]
        customer_id_keywords = ["customer id", "my id", "delivery status", "order status", "track"]

        if any(keyword in user_input.lower() for keyword in cancel_keywords) or any(keyword in user_input.lower() for keyword in customer_id_keywords):
            # Reset the human agent flow
            chat_state["needs_human_agent"] = False
            chat_state["contact_step"] = 0

            # If they're asking about customer ID specifically
            if any(keyword in user_input.lower() for keyword in customer_id_keywords):
                response = "I understand you'd like to check your order status using your customer ID. Could you please provide your customer ID? It should be a 32-character alphanumeric code."
                chat_state["messages"].append({"role": "assistant", "content": response})
                return chat_state
            else:
                response = "I've canceled the request to speak with a human representative. How else can I help you today?"
                chat_state["messages"].append({"role": "assistant", "content": response})
                return chat_state

    # Handle human agent requests
    if intent == "human_agent" and not chat_state.get("needs_human_agent"):
        chat_state["needs_human_agent"] = True
        chat_state["contact_step"] = 0  # Reset step counter
        chat_state["messages"].append({
            "role": "assistant",
            "content": "I'll connect you with a human representative. Could you please provide your name?"
        })
        chat_state["contact_step"] = 1  # Move to next step
        return chat_state

    # Direct handling of contact info collection
    if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
        step = chat_state.get("contact_step", 0)

        if step == 1:  # We asked for name
            chat_state["customer_name"] = user_input
            chat_state["contact_step"] = 2
            chat_state["messages"].append({
                "role": "assistant",
                "content": f"Thank you, {chat_state['customer_name']}. Could you please provide your email address?"
            })
            return chat_state
        elif step == 2:  # We asked for email
            chat_state["customer_email"] = user_input
            chat_state["contact_step"] = 3
            chat_state["messages"].append({
                "role": "assistant",
                "content": "Thank you. Finally, could you please provide your phone number?"
            })
            return chat_state
        elif step == 3:  # We asked for phone
            chat_state["customer_phone"] = user_input
            chat_state["contact_info_collected"] = True
            chat_state["contact_step"] = 4

            # Save contact info to CSV
            success = contact_service.save_contact_info(
                chat_state["customer_name"],
                chat_state["customer_email"],
                chat_state["customer_phone"]
            )

            if success:
                chat_state["messages"].append({
                    "role": "assistant",
                    "content": f"Thank you for providing your information. A customer service representative will contact you soon at {chat_state['customer_email']} or {chat_state['customer_phone']}. Is there anything else you'd like to add before I submit your request?"
                })
            else:
                chat_state["messages"].append({
                    "role": "assistant",
                    "content": "Thank you for providing your information. I encountered an issue saving your contact details, but a customer service representative will be notified. Is there anything specific you'd like me to tell them about your inquiry?"
                })
            return chat_state

    # Direct handling of order lookups with IDs
    if not chat_state.get("order_lookup_attempted") and re.search(r'\b([a-f0-9]{32})\b', user_input):
        id_match = re.search(r'\b([a-f0-9]{32})\b', user_input)
        extracted_id = id_match.group(1)
        logger.info(f"Attempting direct lookup for ID: {extracted_id}")

        try:
            # Try as order ID first
            status, details = order_service.lookup_order_by_id(extracted_id)

            if status:
                # Found as order ID
                response = format_order_details(extracted_id, status, details)
                chat_state["messages"].append({"role": "assistant", "content": response})
                chat_state["order_lookup_attempted"] = True
                chat_state["current_order_id"] = extracted_id
                return chat_state

            # Try as customer ID
            customer_orders = order_service.lookup_orders_by_customer_id(extracted_id)

            if customer_orders:
                if len(customer_orders) == 1:
                    # Only one order for this customer
                    order_data = customer_orders[0]
                    order_id = order_data['order_id']
                    status = order_data['order_status']
                    details = {
                        'purchase_date': order_data['order_purchase_timestamp'],
                        'delivery_date': order_data['order_delivered_customer_date'],
                        'approved_date': order_data['order_approved_at'],
                        'estimated_delivery': order_data['order_estimated_delivery_date'],
                        'actual_delivery': order_data['order_delivered_customer_date']
                    }
                    response = f"I found an order for your customer ID. {format_order_details(order_id, status, details)}"
                else:
                    # Multiple orders for this customer
                    sorted_orders = sorted(customer_orders,
                                        key=lambda x: pd.to_datetime(x['order_purchase_timestamp']),
                                        reverse=True)
                    recent_order = sorted_orders[0]
                    order_id = recent_order['order_id']
                    status = recent_order['order_status']
                    details = {
                        'purchase_date': recent_order['order_purchase_timestamp'],
                        'delivery_date': recent_order['order_delivered_customer_date'],
                        'approved_date': recent_order['order_approved_at'],
                        'estimated_delivery': recent_order['order_estimated_delivery_date'],
                        'actual_delivery': recent_order['order_delivered_customer_date']
                    }
                    response = f"I found {len(customer_orders)} orders for your customer ID. Here's the status of your most recent order: {format_order_details(order_id, status, details)}"

                chat_state["messages"].append({"role": "assistant", "content": response})
                chat_state["order_lookup_attempted"] = True
                chat_state["current_order_id"] = order_id
                return chat_state

            # Not found as either order ID or customer ID
            response = f"I couldn't find any orders with ID {extracted_id}. Please check the ID and try again."
            chat_state["messages"].append({"role": "assistant", "content": response})
            chat_state["order_lookup_attempted"] = True
            return chat_state

        except Exception as e:
            logger.error(f"Direct lookup failed: {e}")
            # Continue with graph-based processing

    # General order status question without specific ID
    if intent == "order_status" and not chat_state.get("order_lookup_attempted"):
        response = """To check your order status, I'll need your order ID or customer ID.

The order ID is a 32-character code that was included in your order confirmation email. It looks something like: e481f51cbdc54678b7cc49136f2d6af7

Could you please provide your order ID?"""
        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Process through chatbot for all other cases
    try:
        chatbot = create_chatbot()
        result = chatbot.invoke(chat_state, config={"recursion_limit": 30})
        return result
    except Exception as e:
        logger.error(f"Error in chatbot processing: {e}")
        chat_state["messages"].append({
            "role": "assistant",
            "content": "I'm sorry, I don't have specific information about that. Would you like to ask about our return policy, shipping options, order status, or speak with a human representative?"
        })
        return chat_state

# ===== CLI Interface =====

if __name__ == "__main__":
    # Simple CLI for testing
    print("E-commerce Support Chatbot (type 'exit' to quit)")
    state = {
        "messages": [],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False,
        "customer_name": None,
        "customer_email": None,
        "customer_phone": None,
        "contact_step": 0
    }

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == "exit":
            break

        state = chat_with_user(user_input, state)
        assistant_message = state["messages"][-1]["content"]
        print(f"\nAssistant: {assistant_message}")

        # Check if conversation has ended
        if state.get("needs_human_agent") and state.get("contact_info_collected"):
            print("\nConversation has been handed off to a human representative.")
            break
