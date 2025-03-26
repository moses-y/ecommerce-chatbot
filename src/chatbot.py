# src/chatbot.py

import os
import sys
import time
import re
import pandas as pd
import logging
import warnings
from functools import lru_cache
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Tuple

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Ensure this path is correct for your structure

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

load_dotenv()

# Assuming these imports are correct relative to src/chatbot.py
from src.config import ORDER_STATUS_DESCRIPTIONS, CONVERSATION_CONFIG, FAQ_CONFIG
# Make sure format_order_details is imported if it's in utils.py
from src.utils import load_order_data, format_order_details, create_order_index
from src.state_management import ConversationMemory
from src.llm_service import LLMService
from src.state_utils import reset_state
from src.vector_db import (
    get_vector_db_instance,
    cached_get_order_by_id,
    cached_get_orders_by_customer_id
)
from src.credentials import verify_credentials

# Initialize conversation memory
conversation_memory = ConversationMemory(
    max_history=CONVERSATION_CONFIG["max_history_length"]
)
FAQ_RESPONSES = FAQ_CONFIG["responses"]

# Verify credentials up front.
try:
    creds = verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
    if not all(creds.values()):
        missing = [k for k, v in creds.items() if not v]
        raise EnvironmentError(f"Missing required credentials: {missing}")
except Exception as exc:
    logger.error(f"Credential verification failed: {exc}")
    raise

@lru_cache(maxsize=1)
def load_order_data_cached():
    # Ensure load_order_data is correctly implemented
    return load_order_data()

class ChatbotState(TypedDict):
    messages: List[Dict[str, Any]]
    order_lookup_attempted: bool
    current_order_id: Optional[str]
    needs_human_agent: bool
    contact_info_collected: bool
    customer_name: Optional[str]
    customer_email: Optional[str]
    customer_phone: Optional[str]
    contact_step: int
    chat_history: List[Dict[str, Any]] # Consider removing if messages is the source of truth
    session_id: str
    feedback: Optional[str]
    type: str # Ensure this is consistently used or removed if not needed

class OrderService:
    """Handles order-related operations."""
    def __init__(self, orders_df=None, vector_collection=None):
        self.orders_df = orders_df if orders_df is not None else load_order_data_cached()
        # Ensure create_order_index handles potential empty DataFrame
        self.order_index, self.customer_index = create_order_index(self.orders_df)
        # Ensure get_vector_db_instance handles potential empty DataFrame
        self.vector_collection = vector_collection if vector_collection is not None else get_vector_db_instance(self.orders_df)
        logger.info(f"OrderService initialized. Order index size: {len(self.order_index)}, Customer index size: {len(self.customer_index)}")


    def lookup_order_by_id(self, order_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Looks up order by Order ID using index first, then vector DB."""
        if order_id in self.order_index:
            order_data = self.order_index[order_id]
            status = order_data.get("order_status")
            # Ensure all expected keys exist in order_data or use .get()
            details = {
                "purchase_date": order_data.get("order_purchase_timestamp"),
                "delivery_date": order_data.get("order_delivered_customer_date"),
                "approved_date": order_data.get("order_approved_at"),
                "estimated_delivery": order_data.get("order_estimated_delivery_date"),
                "actual_delivery": order_data.get("order_delivered_customer_date") # Often same as delivery_date
            }
            logger.info(f"Order {order_id} found in index. Status: {status}")
            return status, details

        # Fallback to vector DB if not in index
        logger.info(f"Order {order_id} not in index, querying vector DB.")
        order_result = cached_get_order_by_id(order_id, self.vector_collection)
        if order_result and isinstance(order_result, dict) and "metadata" in order_result:
            meta = order_result["metadata"]
            status = meta.get("status")
            details = {
                "purchase_date": meta.get("purchase_date"),
                "delivery_date": meta.get("customer_delivery_date"),
                "approved_date": meta.get("approved_date"),
                "estimated_delivery": meta.get("estimated_delivery_date"),
                "actual_delivery": meta.get("customer_delivery_date")
            }
            logger.info(f"Order {order_id} found in vector DB. Status: {status}")
            return status, details
        logger.warning(f"Order {order_id} not found in index or vector DB.")
        return None, None

    def lookup_orders_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        """Looks up orders by Customer ID using index first, then vector DB."""
        if customer_id in self.customer_index:
            orders = self.customer_index[customer_id]
            logger.info(f"Found {len(orders)} orders for customer {customer_id} in index.")
            return orders

        # Fallback to vector DB
        logger.info(f"Customer {customer_id} not in index, querying vector DB.")
        vector_orders = cached_get_orders_by_customer_id(customer_id, self.vector_collection)
        # Ensure the return type is a list of dicts
        if isinstance(vector_orders, list):
             logger.info(f"Found {len(vector_orders)} orders for customer {customer_id} in vector DB.")
             return vector_orders
        logger.warning(f"No orders found for customer {customer_id} in index or vector DB.")
        return []

class ContactService:
    """Handles customer contact information storage."""
    def save_contact_info(self, name: str, email: str, phone: str) -> bool:
        """Saves contact info to a CSV file."""
        try:
            import csv
            # Define path relative to the script or use absolute path if needed
            data_dir = "data"
            os.makedirs(data_dir, exist_ok=True)
            # Ensure filename is clear
            csv_path = os.path.join(data_dir, "human_agent_requests.csv")
            file_exists = os.path.isfile(csv_path)
            # Use 'a+' mode to create if not exists, append otherwise
            with open(csv_path, "a+", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["timestamp", "name", "email", "phone"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                # Write header only if the file is newly created (or empty)
                csvfile.seek(0, os.SEEK_END) # Go to end of file
                if not file_exists or csvfile.tell() == 0:
                    writer.writeheader()
                # Write the data
                writer.writerow({
                    "timestamp": datetime.now().isoformat(),
                    "name": name,
                    "email": email,
                    "phone": phone
                })
            logger.info(f"Contact info saved successfully to {csv_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving contact info to CSV: {e}", exc_info=True)
            return False

# Initialize core services.
try:
    llm_service = LLMService() # Ensure LLMService is correctly implemented
    order_service = OrderService()
    contact_service = ContactService()
    logger.info("Core services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization failed: {e}", exc_info=True)
    # Depending on severity, you might want to raise the exception
    # raise # Uncomment to stop execution if services fail
    sys.exit("Fatal Error: Could not initialize core services.") # Or exit


# --- MODIFIED: lookup_order ---
def lookup_order(state: ChatbotState) -> ChatbotState:
    """Handles order lookup based on provided order/customer ID."""
    new_state = state.copy()
    # Ensure messages list exists and is not empty before accessing the last element
    if not new_state.get("messages"):
        logger.warning("lookup_order called with empty messages state.")
        # Add a default response if called unexpectedly
        new_state["messages"] = new_state.get("messages", []) + [{
            "role": "assistant",
            "content": "I need an order ID or customer ID to look up status. Could you please provide one?"
        }]
        new_state["order_lookup_attempted"] = True # Mark attempt
        return new_state

    message = new_state["messages"][-1]["content"]
    # Relax regex slightly to allow potential leading/trailing spaces if needed, but keep core pattern
    match = re.search(r'\b([a-f0-9]{32})\b', message)

    if match:
        extracted_id = match.group(1)
        logger.info(f"Attempting lookup with ID: {extracted_id}")
        status, details = order_service.lookup_order_by_id(extracted_id)

        if status and details: # Found by Order ID
            logger.info(f"Order found by Order ID {extracted_id}. Status: {status}")
            # Pass the actual details dictionary to format_order_details
            # Ensure format_order_details handles potential None values in details gracefully
            response = format_order_details(extracted_id, status, details)
            new_state["current_order_id"] = extracted_id # Store the specific order ID found

        else: # Try searching by Customer ID
            logger.info(f"Order ID {extracted_id} not found, trying as Customer ID.")
            orders = order_service.lookup_orders_by_customer_id(extracted_id)
            if orders: # Ensure orders is a non-empty list
                if len(orders) == 1:
                    order_data = orders[0]
                    order_id = order_data.get("order_id", "N/A") # Use .get() for safety
                    status = order_data.get("order_status", "N/A")
                    # Reconstruct details dict for format_order_details
                    order_details = {
                        "purchase_date": order_data.get("order_purchase_timestamp"),
                        "delivery_date": order_data.get("order_delivered_customer_date"),
                        "approved_date": order_data.get("order_approved_at"),
                        "estimated_delivery": order_data.get("order_estimated_delivery_date"),
                        "actual_delivery": order_data.get("order_delivered_customer_date")
                    }
                    logger.info(f"Found 1 order for Customer ID {extracted_id}. Order ID: {order_id}, Status: {status}")
                    response = f"I found one order associated with your customer ID. {format_order_details(order_id, status, order_details)}"
                    new_state["current_order_id"] = order_id # Store the specific order ID found

                else:
                    # Sort orders by purchase date, most recent first
                    try:
                        sorted_orders = sorted(
                            orders,
                            # Provide a default far past/future date if timestamp is missing/invalid for robust sorting
                            key=lambda x: pd.to_datetime(x.get("order_purchase_timestamp", pd.Timestamp.min)),
                            reverse=True
                        )
                    except Exception as sort_e:
                         logger.error(f"Error sorting orders for customer {extracted_id}: {sort_e}")
                         # Fallback: use the first order found if sorting fails
                         sorted_orders = orders

                    if not sorted_orders: # Should not happen if orders was non-empty, but check anyway
                         logger.error(f"Order list became empty after sorting attempt for customer {extracted_id}")
                         response = f"I found multiple orders for customer ID {extracted_id}, but encountered an issue retrieving details."
                    else:
                         recent_order = sorted_orders[0]
                         order_id = recent_order.get("order_id", "N/A")
                         status = recent_order.get("order_status", "N/A")
                         # Reconstruct details dict for format_order_details
                         recent_order_details = {
                             "purchase_date": recent_order.get("order_purchase_timestamp"),
                             "delivery_date": recent_order.get("order_delivered_customer_date"),
                             "approved_date": recent_order.get("order_approved_at"),
                             "estimated_delivery": recent_order.get("order_estimated_delivery_date"),
                             "actual_delivery": recent_order.get("order_delivered_customer_date")
                         }
                         logger.info(f"Found {len(orders)} orders for Customer ID {extracted_id}. Displaying most recent: Order ID: {order_id}, Status: {status}")
                         response = (f"I found {len(orders)} orders associated with your customer ID. "
                                     f"Here's the status of your most recent one: "
                                     f"{format_order_details(order_id, status, recent_order_details)}")
                         new_state["current_order_id"] = order_id # Store the most recent order ID

            else: # ID not found as Order ID or Customer ID
                logger.warning(f"Could not find any orders matching ID: {extracted_id}")
                response = f"I couldn't find any orders associated with the ID '{extracted_id}'. Please double-check the ID and try again. It should be a 32-character code containing letters (a-f) and numbers (0-9)."

        # Append the determined response
        new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response}]
        new_state["order_lookup_attempted"] = True # Mark attempt successful or not

    else: # No 32-character ID found in the message
        logger.info("No valid 32-character ID found in the message for lookup.")
        response = ("I'd be happy to help check your order status. "
                    "Could you please provide your order ID or customer ID? "
                    "It should be a 32-character alphanumeric code (letters a-f, numbers 0-9).")
        new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response}]
        # Mark that we prompted for an ID, preventing immediate re-prompting if the user says something else next
        new_state["order_lookup_attempted"] = True

    return new_state


# --- collect_contact_info ---
# (Keep this function as it was, it seems correct for gathering info step-by-step
# and saving to CSV via ContactService)
def collect_contact_info(state: ChatbotState) -> ChatbotState:
    """Collects customer information for human support."""
    new_state = state.copy()
    # Ensure messages list exists and is not empty
    if not new_state.get("messages"):
        logger.warning("collect_contact_info called with empty messages state.")
        # Maybe add a message indicating the start?
        new_state["messages"] = new_state.get("messages", []) + [{
            "role": "assistant",
            "content": "It looks like you want to speak to a human. Let's collect your contact details."
        }]
        # Ensure contact_step is initialized if starting fresh
        if "contact_step" not in new_state:
             new_state["contact_step"] = 0

    # Get last user message safely
    last_user_message_content = ""
    if new_state["messages"] and new_state["messages"][-1].get("role") == "user":
        last_user_message_content = new_state["messages"][-1].get("content", "").strip()

    lower_msg = last_user_message_content.lower()

    # Cancellation logic
    cancel_words = ["cancel", "nevermind", "never mind", "stop", "go back", "forget it"]
    id_words = ["customer id", "my id", "delivery status", "order status", "track", "check order"] # Added more context
    if any(word in lower_msg for word in cancel_words) or (any(word in lower_msg for word in id_words) and new_state.get("contact_step", 0) > 0):
        logger.info("User cancelled contact collection.")
        new_state["needs_human_agent"] = False
        new_state["contact_info_collected"] = False # Ensure this is reset
        new_state["contact_step"] = 0
        # Clear potentially collected partial info
        new_state["customer_name"] = None
        new_state["customer_email"] = None
        new_state["customer_phone"] = None

        if any(word in lower_msg for word in id_words):
            response_content = ("Okay, I've canceled the request for a human agent. "
                                "To check your order status instead, please provide your order ID or customer ID (a 32-character alphanumeric code).")
            # Set flag to indicate we are now prompting for ID
            new_state["order_lookup_attempted"] = True
        else:
            response_content = "Okay, I've canceled the request to speak with a human representative. How else can I help you today?"

        new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response_content}]
        return new_state

    # Contact collection steps
    step = new_state.get("contact_step", 0)

    # Check if the last message was from the assistant; if so, wait for user input
    if new_state["messages"] and new_state["messages"][-1].get("role") == "assistant":
         logger.debug("Waiting for user input during contact collection.")
         return new_state # Don't proceed if the last message wasn't user input for steps > 0

    response_content = ""
    if step == 0: # Initial trigger (should ideally be set by router)
        new_state["contact_step"] = 1
        response_content = "Okay, I can help connect you with a human representative. First, could you please provide your full name?"
    elif step == 1: # Received name
        if not last_user_message_content:
             response_content = "Sorry, I didn't catch that. Could you please provide your full name?"
        else:
             new_state["customer_name"] = last_user_message_content
             new_state["contact_step"] = 2
             response_content = f"Thank you, {last_user_message_content}. Next, could you please provide your email address?"
    elif step == 2: # Received email
        # Basic email validation (optional but recommended)
        if not last_user_message_content or "@" not in last_user_message_content or "." not in last_user_message_content:
             response_content = "That doesn't look like a valid email address. Could you please provide your email again?"
        else:
             new_state["customer_email"] = last_user_message_content
             new_state["contact_step"] = 3
             response_content = "Great, thank you. Finally, could you please provide your phone number?"
    elif step == 3: # Received phone
        # Basic phone validation (optional) - very simple check
        if not last_user_message_content or not any(char.isdigit() for char in last_user_message_content):
             response_content = "Sorry, that doesn't seem like a valid phone number. Could you please provide it again?"
        else:
             new_state["customer_phone"] = last_user_message_content
             new_state["contact_info_collected"] = True
             new_state["contact_step"] = 4 # Mark as completed
             logger.info(f"Collected contact info: Name={new_state['customer_name']}, Email={new_state['customer_email']}, Phone={new_state['customer_phone']}")
             # Attempt to save contact info
             save_success = contact_service.save_contact_info(
                 new_state["customer_name"], new_state["customer_email"], new_state["customer_phone"]
             )
             if save_success:
                 response_content = (f"Thank you! I've recorded your information. A customer service representative will contact you soon at "
                                     f"{new_state['customer_email']} or {new_state['customer_phone']}. Is there anything else I can help you with in the meantime?")
             else:
                 # Inform user if saving failed, but proceed as if collected
                 response_content = (f"Thank you for providing your information ({new_state['customer_email']} / {new_state['customer_phone']}). "
                                     f"I had trouble saving it to our system, but please wait, and a representative should still reach out. "
                                     f"Is there anything else I can help with?")
             # Reset needs_human_agent? Depends on desired flow after collection.
             # new_state["needs_human_agent"] = False # Optional: If bot should continue normally after collection

    # Append the response if one was generated
    if response_content:
         new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response_content}]

    return new_state


# --- continue_conversation ---
# (Keep this function as is, it handles the general LLM call)
def continue_conversation(state: ChatbotState) -> ChatbotState:
    """Generates a response using the LLM service."""
    new_state = state.copy()
    # Ensure messages list exists
    if not new_state.get("messages"):
         logger.warning("continue_conversation called with empty messages state.")
         # Provide a default greeting or prompt
         new_state["messages"] = [{"role": "assistant", "content": "Hello! How can I help you today?"}]
         return new_state

    logger.info("Generating LLM response via continue_conversation.")
    # Add context from state if needed for LLM prompt engineering here
    # For example, pass relevant parts of the state to llm_service
    try:
        # Pass only messages, let llm_service handle memory if needed internally
        response_text = llm_service.generate_response(new_state["messages"], conversation_memory)
        # Basic check for empty response
        if not response_text or not response_text.strip():
             logger.warning("LLM service returned empty response.")
             response_text = "I'm sorry, I couldn't generate a response for that. Could you try rephrasing?"

        new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response_text}]
    except Exception as llm_e:
        logger.error(f"Error during LLM response generation: {llm_e}", exc_info=True)
        # Provide a fallback response
        fallback_response = "I'm currently experiencing difficulties generating a response. Please try again shortly."
        new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": fallback_response}]

    return new_state


# --- connect_to_agent ---
# (This node seems unused based on the router logic, which uses collect_contact_info.
# If you want an immediate transfer *without* collecting info, you'd need a different route.
# Keeping it for now, but it might be dead code.)
def connect_to_agent(state: ChatbotState) -> ChatbotState:
    """Immediately transfer the conversation to a human representative."""
    new_state = state.copy()
    logger.info("Executing connect_to_agent node (Immediate Transfer).")
    response_content = "I'm connecting you to a human representative now. Please hold while I transfer your chat."
    new_state["messages"] = new_state.get("messages", []) + [{"role": "assistant", "content": response_content}]
    new_state["needs_human_agent"] = True # Flag that transfer is initiated
    new_state["contact_info_collected"] = True # Mark as 'collected' to satisfy router/end conditions
    # Add actual transfer logic here if integrating with a live chat system
    return new_state


# --- MODIFIED: router ---
def router(state: ChatbotState) -> str:
    """Determines the next node based on the last message and state."""
    # Ensure messages list exists and is not empty
    if not state.get("messages"):
         logger.warning("Router called with empty messages state.")
         return END # Or some default error state/node

    # Get last message details safely
    last_message = state["messages"][-1]
    last_message_content = last_message.get("content", "")
    last_message_role = last_message.get("role", "")
    lower_msg = last_message_content.lower()

    logger.debug(f"Router evaluating state. Last message role: {last_message_role}, content: '{last_message_content[:50]}...', order_lookup_attempted: {state.get('order_lookup_attempted')}, needs_human_agent: {state.get('needs_human_agent')}, contact_step: {state.get('contact_step')}")

    # --- Flow Control Logic ---

    # 1. Human Agent Request / Collection Flow
    human_words = ["speak to a human", "talk to a human", "human representative",
                   "real person", "speak to an agent", "talk to a representative",
                   "connect me with a human", "human agent please", "agent", "representative", "live agent"]
    # Check if user explicitly asks for human *now* and collection hasn't started/finished
    if last_message_role == "user" and any(word in lower_msg for word in human_words) and not state.get("needs_human_agent"):
        logger.info("Routing to collect_contact_info (user requested human).")
        state["needs_human_agent"] = True # Set the flag
        state["contact_info_collected"] = False # Ensure not marked collected yet
        state["contact_step"] = 0 # Reset/initialize step
        return "collect_contact_info"

    # Check if collection is in progress (needs agent flag is set, not yet collected)
    if state.get("needs_human_agent") and not state.get("contact_info_collected"):
        # Only proceed if the last message was from the user (providing info)
        if last_message_role == "user":
             logger.info("Routing to collect_contact_info (collection in progress).")
             return "collect_contact_info"
        else:
             # If last message was assistant's prompt, wait for user's reply
             logger.debug("Router waiting for user input during contact collection.")
             return END # End this graph iteration, wait for next user input

    # 2. End Conversation Keywords
    end_words = ["bye", "goodbye", "exit", "quit", "that's all", "no thanks"]
    if last_message_role == "user" and any(term in lower_msg for term in end_words):
         logger.info("Routing to END (user ended conversation).")
         return END

    # 3. Order Lookup Flow
    has_id = bool(re.search(r"\b([a-f0-9]{32})\b", lower_msg))
    order_query_words = ["order status", "track my order", "where is my order", "delivery status"]

    # If user provides an ID
    if last_message_role == "user" and has_id:
         # Allow lookup even if attempted before, user might provide correct ID now
         state["order_lookup_attempted"] = False # Reset flag to allow lookup
         logger.info("Routing to lookup_order (user provided ID).")
         return "lookup_order"

    # If user asks about order status *without* providing an ID
    if last_message_role == "user" and any(phrase in lower_msg for phrase in order_query_words) and not has_id:
         # Check if we already prompted in the *immediately preceding* turn
         if len(state["messages"]) > 1 and state["messages"][-2].get("role") == "assistant" and "order id or customer id" in state["messages"][-2].get("content", "").lower():
              logger.info("User asked for status again without ID after prompt, routing to continue_conversation.")
              return "continue_conversation" # Avoid immediate re-prompt, let LLM handle
         else:
              logger.info("Routing to lookup_order (user asked for status without ID, will prompt).")
              # Route to lookup_order, its 'else' block will handle the prompt
              return "lookup_order"

    # 4. Default to General Conversation
    # Only route to LLM if the last message was from the user
    if last_message_role == "user":
        logger.info("Routing to continue_conversation (default).")
        return "continue_conversation"
    else:
        # If last message was from assistant, end this iteration and wait for user
        logger.debug("Router waiting for user input (default).")
        return END


# --- create_chatbot ---
# (Ensure this function correctly adds nodes and edges using the modified functions)
def create_chatbot():
    """Compiles the chatbot workflow using a state graph."""
    try:
        # Verify credentials again? Optional, depends if they can expire mid-session
        # verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])

        workflow = StateGraph(ChatbotState)

        # Add nodes to the graph
        workflow.add_node("continue_conversation", continue_conversation)
        workflow.add_node("lookup_order", lookup_order)
        workflow.add_node("collect_contact_info", collect_contact_info)
        # workflow.add_node("connect_to_agent", connect_to_agent) # Add if used

        # Set the entry point
        # The entry point should ideally just receive the user input and pass to router,
        # but LangGraph often uses a default node like continue_conversation.
        # Let's assume the first user message triggers the flow.
        workflow.set_entry_point("router_entry") # Use a dedicated entry router

        # Define an entry router function that decides the first real step
        def router_entry(state: ChatbotState) -> str:
             # This function runs *once* at the beginning of an invoke call
             # It should look at the *latest* user message added in chat_with_user
             logger.debug("Executing router_entry")
             # Delegate the actual routing logic to the main router function
             return router(state)

        # Add the entry router node
        workflow.add_node("router_entry", router_entry)


        # Define edges based on the router's output
        # Edges originate from the nodes that can lead to a choice point.
        # Since our router runs after each step, we add conditional edges from all main nodes.

        # Conditional edges based on the output of the main router function
        # The router function itself is NOT a node, but its logic determines the transition *after* a node completes.
        # We need conditional edges *from* each node that might require routing.

        # Option 1: Simple - All nodes lead back to the router_entry (less efficient)
        # workflow.add_edge("continue_conversation", "router_entry")
        # workflow.add_edge("lookup_order", "router_entry")
        # workflow.add_edge("collect_contact_info", "router_entry")

        # Option 2: More explicit conditional edges (Preferred)
        # After a node runs, the state is updated. The conditional edge then calls the router
        # function with the *updated* state to decide the *next* node.

        workflow.add_conditional_edges(
            "continue_conversation", # Source node
            router, # Function to call to decide next step
            { # Mapping: output of router -> destination node or END
                "lookup_order": "lookup_order",
                "collect_contact_info": "collect_contact_info",
                "continue_conversation": "continue_conversation", # Allows looping back if needed
                END: END
            }
        )
        workflow.add_conditional_edges(
            "lookup_order",
            router,
            {
                "lookup_order": "lookup_order", # Should be less common now
                "collect_contact_info": "collect_contact_info",
                "continue_conversation": "continue_conversation",
                END: END
            }
        )
        workflow.add_conditional_edges(
            "collect_contact_info",
            router,
            {
                "lookup_order": "lookup_order",
                "collect_contact_info": "collect_contact_info", # Continue collection
                "continue_conversation": "continue_conversation", # If collection finished/cancelled
                END: END
            }
        )

        # Compile the graph
        logger.info("Compiling the state graph.")
        chatbot_graph = workflow.compile()
        logger.info("State graph compiled successfully.")
        return chatbot_graph

    except Exception as e:
        logger.error(f"Chatbot graph creation failed: {e}", exc_info=True)
        raise # Re-raise the exception to prevent using a broken graph


# --- MODIFIED: chat_with_user ---
# (Ensure this uses the refactored logic from the previous response,
# primarily calling the graph)
def chat_with_user(user_input: str, chat_state: Optional[Dict] = None) -> Dict:
    """Updates the chatbot state based on a user input using the state graph."""
    try:
        # Initialize or get current state
        chat_state = chat_state if chat_state is not None else reset_state()
        logger.debug(f"Input: '{user_input}'. Initial state keys: {list(chat_state.keys())}")

        # Sanitize input
        user_input_cleaned = user_input.strip()
        if not user_input_cleaned:
             logger.warning("Received empty user input.")
             # Return state with a message asking for input
             chat_state["messages"] = chat_state.get("messages", []) + [{"role": "assistant", "content": "Did you mean to say something?"}]
             return chat_state

        # Ensure user message is added only once to the state's message list
        messages = chat_state.get("messages", [])
        # Check if messages list is empty OR if the last message is different from the current input
        if not messages or messages[-1].get("content") != user_input_cleaned or messages[-1].get("role") != "user":
             messages.append({"role": "user", "content": user_input_cleaned})
             chat_state["messages"] = messages # Update state with the new user message

        # --- Intent detection for simple FAQs (handled outside graph) ---
        intent = detect_intent(user_input_cleaned)

        # --- MODIFIED Direct FAQ Handling ---
        # Handle simple FAQs directly, EXCLUDING intents managed by the graph nodes
        graph_handled_intents = ["human_agent", "order_status", "unknown", "affirmation", "negation"] # Intents definitely needing graph logic
        # Check if the detected intent is a simple FAQ response AND NOT explicitly handled by the graph
        if intent and intent in FAQ_RESPONSES and intent not in graph_handled_intents:
            logger.info(f"Handling simple FAQ/response intent directly: {intent}")
            chat_state["messages"].append({"role": "assistant", "content": FAQ_RESPONSES[intent]})
            # If the intent is 'goodbye', we might consider ending the session,
            # but for now, just returning the state with the goodbye message is sufficient.
            return chat_state # Return early for simple FAQs/responses
        # --- END MODIFIED Direct FAQ Handling ---


        # --- Handle cancellation *during* contact info collection ---
        # This logic runs before invoking the graph if a cancellation is detected mid-flow.
        if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
            cancel_terms = ["cancel", "nevermind", "never mind", "stop", "go back", "different question", "forget it"]
            id_terms = ["customer id", "my id", "delivery status", "order status", "track", "check order"]
            lower_input = user_input_cleaned.lower()
            # Check if user input contains cancellation terms OR order-related terms (implying they changed their mind)
            if any(term in lower_input for term in cancel_terms) or (any(term in lower_input for term in id_terms) and chat_state.get("contact_step", 0) > 0):
                logger.info("User cancelled human agent request during collection.")
                # Reset relevant state variables
                chat_state["needs_human_agent"] = False
                chat_state["contact_info_collected"] = False
                chat_state["contact_step"] = 0 # Reset contact state
                chat_state["customer_name"] = None # Clear potentially collected partial info
                chat_state["customer_email"] = None
                chat_state["customer_phone"] = None

                # Determine the appropriate cancellation response
                if any(term in lower_input for term in id_terms):
                     response_content = ("Okay, I've canceled the request for a human agent. "
                                         "To check your order status instead, please provide your order ID or customer ID (a 32-character alphanumeric code).")
                     # Set flag indicating we prompted for an ID, relevant for the next router decision
                     chat_state["order_lookup_attempted"] = True
                else:
                    response_content = "Okay, I've canceled the request to speak with a human representative. How else can I help you today?"

                # Add the cancellation response to the messages
                chat_state["messages"].append({"role": "assistant", "content": response_content})
                return chat_state # Return state immediately after handling cancellation

        # --- Invoke the state graph for all other conversation flow ---
        try:
            logger.info("Invoking state graph...")
            # Get the compiled graph (using the cached version)
            chatbot_graph = get_compiled_graph()

            # Execute the graph with the current state
            # Use a reasonable recursion limit to prevent infinite loops
            result_state = chatbot_graph.invoke(chat_state, config={"recursion_limit": 15})

            # --- Post-graph processing and validation ---
            # Ensure the graph returned a valid dictionary state
            if isinstance(result_state, dict) and "messages" in result_state:
                 logger.debug(f"Graph returned state keys: {list(result_state.keys())}")
                 final_messages = result_state.get("messages", [])
                 # Simple deduplication: remove last message if identical to second-to-last
                 # This can sometimes happen with certain graph flows.
                 if len(final_messages) >= 2 and final_messages[-1] == final_messages[-2]:
                      logger.warning("Graph produced duplicate last message, removing.")
                      final_messages.pop()
                      result_state["messages"] = final_messages # Update state

                 # Ensure core state keys exist (optional but good practice for robustness)
                 result_state.setdefault("order_lookup_attempted", False)
                 result_state.setdefault("needs_human_agent", False)
                 result_state.setdefault("contact_info_collected", False)
                 result_state.setdefault("contact_step", 0)
                 # Ensure session_id persists
                 result_state.setdefault("session_id", chat_state.get("session_id", reset_state()["session_id"]))


                 return result_state
            else:
                 # Log an error if the graph returned something unexpected
                 logger.error(f"Graph invocation returned unexpected result type: {type(result_state)}. Result: {result_state}")
                 # Append a generic error message for the user
                 chat_state["messages"].append({
                     "role": "assistant",
                     "content": "I encountered an internal issue processing that. Could you please try rephrasing?"
                 })
                 return chat_state

        except Exception as graph_e:
            # Handles graph execution errors (e.g., recursion limits, errors within nodes)
            logger.error(f"Chatbot graph processing error: {graph_e}", exc_info=True)
            fallback_message = ("I seem to be having trouble handling that request right now. "
                                "You could try asking differently, or ask about our return policy, shipping options, or request to speak with a human representative.")
            # Avoid adding duplicate error messages if the error persists
            if not chat_state.get("messages") or chat_state["messages"][-1].get("content") != fallback_message:
                 chat_state["messages"] = chat_state.get("messages", []) + [{"role": "assistant", "content": fallback_message}]
            return chat_state

    # --- Top-level exception handling ---
    except EnvironmentError as ee:
        # Handle missing credentials or other environment issues
        logger.error(f"Credential or Environment error in chat_with_user: {ee}", exc_info=True)
        chat_state = chat_state or reset_state() # Ensure state exists
        chat_state["messages"] = chat_state.get("messages", []) + [{
            "role": "assistant",
            "content": "I'm having trouble accessing necessary services due to a configuration issue. Please try again later or contact support."
        }]
        return chat_state
    except Exception as ex:
        # Catch any other unexpected errors during processing
        logger.error(f"Unexpected general error in chat_with_user: {ex}", exc_info=True)
        chat_state = chat_state or reset_state() # Ensure state exists
        generic_error = "I apologize for the technical difficulty. Could you please try asking your question again?"
        # Avoid adding duplicate generic error messages
        if not chat_state.get("messages") or chat_state["messages"][-1].get("content") != generic_error:
             chat_state["messages"] = chat_state.get("messages", []) + [{"role": "assistant", "content": generic_error}]
        return chat_state

# Helper function to get compiled graph (implement caching)
@lru_cache(maxsize=1)
def get_compiled_graph():
    """Creates and compiles the graph, cached for efficiency."""
    logger.info("Compiling graph (cached)...")
    return create_chatbot()


# --- detect_intent ---

def detect_intent(message: str) -> Optional[str]:
    """Detects user intent based on keywords and patterns using prioritized checks."""
    print("DETECT_INTENT_VERSION_CHECK_V9")  # Increment again
    lower_msg = message.lower()
    logger.debug(f"detect_intent input: '{message}', lower: '{lower_msg}'")

    # Use regex to find the ID, store the match object
    id_match = re.search(r"[a-f0-9]{32}", lower_msg)  # Simplified ID regex
    has_id = bool(id_match)
    logger.debug(f"ID found: {has_id}")

    # --- PRIORITY 1: Order Status ---
    # Updated order query keywords list to include 'track my package'
    order_query_words = [
        "check order status", "order status", "track my order", "track my package",
        "where is my order", "delivery status", "check my order", "order", "order id"
    ]

    # --- VERY SPECIFIC DEBUGGING ---
    pattern_to_test_1 = r"\border\b.*?[a-f0-9]{32}.*?\bstatus\b"  # Simplified regex
    pattern_to_test_2 = r"\border status\b.*?[a-f0-9]{32}"  # New regex for "order status ID"
    search_result_1 = re.search(pattern_to_test_1, lower_msg)
    search_result_2 = re.search(pattern_to_test_2, lower_msg)
    print(f"DEBUG: Regex '{pattern_to_test_1}' on '{lower_msg}' -> Result: {search_result_1}")
    print(f"DEBUG: Regex '{pattern_to_test_2}' on '{lower_msg}' -> Result: {search_result_2}")
    # --- END SPECIFIC DEBUGGING ---

    # Specific check using regex for "order" and "status" as whole words, plus the ID.
    if has_id and (search_result_1 or search_result_2 or 
                   re.search(r"\bstatus\b.*?[a-f0-9]{32}.*?\border\b", lower_msg) or 
                   re.search(r"\border status\b.*?[a-f0-9]{32}", lower_msg)):
        logger.debug("Detected intent: order_status (specific regex pattern)")
        return "order_status"

    # General check for common order query phrases (exact matches)
    if any(phrase == lower_msg for phrase in order_query_words):
         logger.debug("Detected intent: order_status (exact query phrase)")
         return "order_status"
    # General check for common order query phrases (substring matches)
    if any(phrase in lower_msg for phrase in order_query_words):
        logger.debug("Detected intent: order_status (substring query phrase)")
        return "order_status"

    # General check if ID is present and the word "order" is also present as a whole word
    if has_id and re.search(r"\border\b", lower_msg):
        logger.debug("Detected intent: order_status (ID + 'order' word)")
        return "order_status"
    # --- END PRIORITY 1 ---

    # --- PRIORITY 2: Human Agent ---
    human_keywords = [
        "speak to a human", "talk to a human", "human representative",
        "real person", "speak to an agent", "talk to a representative",
        "connect me with a human", "human agent please", "agent", "representative", "live agent"
    ]
    if any(word in lower_msg for word in human_keywords):
        logger.debug("Detected intent: human_agent")
        return "human_agent"
    # --- END PRIORITY 2 ---

    # --- PRIORITY 3: General FAQs (from config) ---
    if FAQ_CONFIG and "intent_patterns" in FAQ_CONFIG:
        for intent, patterns in FAQ_CONFIG["intent_patterns"].items():
            if isinstance(patterns, list) and any(pattern in lower_msg for pattern in patterns):
                if intent not in ["order_status", "human_agent"]:
                    logger.debug(f"Detected FAQ intent: {intent}")
                    return intent
    else:
        logger.warning("FAQ_CONFIG is missing or not structured correctly. Skipping FAQ intent detection.")
    # --- END PRIORITY 3 ---

    # --- PRIORITY 4: Greetings & Goodbye ---
    greeting_words = ["hello", "hi", "hey", "greetings"]
    if any(re.search(r"\b" + word + r"\b", lower_msg) for word in greeting_words):
         logger.debug("Detected intent: greeting")
         return "greeting"

    goodbye_words = ["bye", "goodbye", "see you", "later", "exit", "quit", "that's all", "no thanks"]
    if any(re.search(r"\b" + word + r"\b", lower_msg) for word in goodbye_words):
         logger.debug("Detected intent: goodbye")
         return "goodbye"
    # --- END PRIORITY 4 ---

    # --- Fallback: Check if only an ID was provided ---
    if has_id:
        logger.debug("Detected only an ID, no specific intent matched.")
        return None

    logger.debug("No specific intent detected, returning None.")
    return None

if __name__ == "__main__":
    try:
        # Ensure credentials are valid before starting
        verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
        print("E-commerce Support Chatbot (type 'exit' to quit)")
        # Initialize state for the session
        current_chat_state = reset_state()
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                print("Assistant: Goodbye!")
                break
            # Process input and update state
            current_chat_state = chat_with_user(user_input, current_chat_state)
            # Display the latest assistant message
            if current_chat_state.get("messages"):
                print(f"\nAssistant: {current_chat_state['messages'][-1]['content']}")
            else:
                print("\nAssistant: (No response generated)") # Should not happen ideally

            # Check if conversation should end (e.g., after human handoff)
            # This condition might need adjustment based on exact flow
            if current_chat_state.get("needs_human_agent") and current_chat_state.get("contact_info_collected"):
                # Check if the last message confirms handoff
                if "representative will contact you soon" in current_chat_state.get("messages", [])[-1].get("content", ""):
                     print("\n(Contact information collected. A representative will be in touch.)")
                     # Decide if the bot should stop or continue
                     # break # Uncomment to end chat after successful collection
    except Exception as e:
        logger.error(f"Chatbot startup or runtime error: {e}", exc_info=True)
        print(f"Error: Could not run chatbot. Details: {e}")
