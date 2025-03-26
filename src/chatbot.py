# src/chatbot.py
"""
E-commerce support chatbot using a state graph for conversation management.
Implements a hexagonal architecture with clear separation of concerns.
"""

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
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

from src.config import ORDER_STATUS_DESCRIPTIONS, CONVERSATION_CONFIG, FAQ_CONFIG
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
    chat_history: List[Dict[str, Any]]
    session_id: str
    feedback: Optional[str]
    type: str

class OrderService:
    """Handles order-related operations."""
    def __init__(self, orders_df=None, vector_collection=None):
        self.orders_df = orders_df or load_order_data_cached()
        self.order_index, self.customer_index = create_order_index(self.orders_df)
        self.vector_collection = vector_collection or get_vector_db_instance(self.orders_df)

    def lookup_order_by_id(self, order_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        if order_id in self.order_index:
            order_data = self.order_index[order_id]
            status = order_data["order_status"]
            details = {
                "purchase_date": order_data["order_purchase_timestamp"],
                "delivery_date": order_data["order_delivered_customer_date"],
                "approved_date": order_data["order_approved_at"],
                "estimated_delivery": order_data["order_estimated_delivery_date"],
                "actual_delivery": order_data["order_delivered_customer_date"]
            }
            return status, details
        order_result = cached_get_order_by_id(order_id, self.vector_collection)
        if order_result:
            meta = order_result["metadata"]
            status = meta["status"]
            details = {
                "purchase_date": meta["purchase_date"],
                "delivery_date": meta.get("customer_delivery_date"),
                "approved_date": meta.get("approved_date"),
                "estimated_delivery": meta.get("estimated_delivery_date"),
                "actual_delivery": meta.get("customer_delivery_date")
            }
            return status, details
        return None, None

    def lookup_orders_by_customer_id(self, customer_id: str) -> List[Dict[str, Any]]:
        if customer_id in self.customer_index:
            return self.customer_index[customer_id]
        return cached_get_orders_by_customer_id(customer_id, self.vector_collection)

class ContactService:
    """Handles customer contact information storage."""
    def save_contact_info(self, name: str, email: str, phone: str) -> bool:
        try:
            import csv
            os.makedirs("data", exist_ok=True)
            csv_path = os.path.join("data", "contact_requests.csv")
            file_exists = os.path.isfile(csv_path)
            with open(csv_path, "a", newline="") as csvfile:
                fieldnames = ["name", "email", "phone"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({"name": name, "email": email, "phone": phone})
            return True
        except Exception as e:
            logger.error(f"Error saving contact info: {e}")
            return False

# Initialize core services.
try:
    llm_service = LLMService()
    order_service = OrderService()
    contact_service = ContactService()
    logger.info("Core services initialized successfully")
except Exception as e:
    logger.error(f"Service initialization failed: {e}")
    raise

def lookup_order(state: ChatbotState) -> ChatbotState:
    """Handles order lookup based on provided order/customer ID."""
    new_state = state.copy()
    message = state["messages"][-1]["content"]
    match = re.search(r'\b([a-f0-9]{32})\b', message)
    if match:
        extracted_id = match.group(1)
        status, details = order_service.lookup_order_by_id(extracted_id)
        if status:
            response = format_order_details(extracted_id, status, details)
        else:
            orders = order_service.lookup_orders_by_customer_id(extracted_id)
            if orders:
                if len(orders) == 1:
                    order_data = orders[0]
                    order_id = order_data["order_id"]
                    status = order_data["order_status"]
                    details = {
                        "purchase_date": order_data["order_purchase_timestamp"],
                        "delivery_date": order_data["order_delivered_customer_date"],
                        "approved_date": order_data["order_approved_at"],
                        "estimated_delivery": order_data["order_estimated_delivery_date"],
                        "actual_delivery": order_data["order_delivered_customer_date"]
                    }
                    response = f"I found an order for your customer ID. {format_order_details(order_id, status, details)}"
                else:
                    sorted_orders = sorted(
                        orders,
                        key=lambda x: pd.to_datetime(x["order_purchase_timestamp"]),
                        reverse=True
                    )
                    recent_order = sorted_orders[0]
                    order_id = recent_order["order_id"]
                    status = recent_order["order_status"]
                    details = {
                        "purchase_date": recent_order["order_purchase_timestamp"],
                        "delivery_date": recent_order["order_delivered_customer_date"],
                        "approved_date": recent_order["order_approved_at"],
                        "estimated_delivery": recent_order["order_estimated_delivery_date"],
                        "actual_delivery": recent_order["order_delivered_customer_date"]
                    }
                    response = f"I found {len(orders)} orders for your customer ID. Here's the status of your most recent order: {format_order_details(order_id, status, details)}"
            else:
                response = f"I couldn't find any orders with ID {extracted_id}. Please check and try again."
        new_state["messages"].append({"role": "assistant", "content": response})
        new_state["order_lookup_attempted"] = True
        new_state["current_order_id"] = extracted_id
    else:
        response = ("I'd be happy to help check your order status. "
                    "Could you please provide your order ID or customer ID? "
                    "It should be a 32-character alphanumeric code.")
        new_state["messages"].append({"role": "assistant", "content": response})
    return new_state

def collect_contact_info(state: ChatbotState) -> ChatbotState:
    """Collects customer information for human support."""
    new_state = state.copy()
    last_message = state["messages"][-1]["content"].strip()
    lower_msg = last_message.lower()
    cancel_words = ["cancel", "nevermind", "never mind", "stop", "go back"]
    id_words = ["customer id", "my id", "delivery status", "order status", "track"]
    if any(word in lower_msg for word in cancel_words) or any(word in lower_msg for word in id_words):
        new_state["needs_human_agent"] = False
        new_state["contact_step"] = 0
        if any(word in lower_msg for word in id_words):
            new_state["messages"].append({
                "role": "assistant",
                "content": ("I understand you'd like to check your order status using your customer ID. "
                            "Could you please provide your customer ID? It should be a 32-character alphanumeric code.")
            })
        else:
            new_state["messages"].append({
                "role": "assistant",
                "content": "I've canceled the request to speak with a human representative. How else can I help you?"
            })
        return new_state

    step = new_state.get("contact_step", 0)
    if step == 0:
        new_state["contact_step"] = 1
        new_state["messages"].append({
            "role": "assistant",
            "content": "I'll connect you with a human representative. Could you please provide your name?"
        })
    elif step == 1:
        new_state["customer_name"] = last_message
        new_state["contact_step"] = 2
        new_state["messages"].append({
            "role": "assistant",
            "content": f"Thank you, {last_message}. Could you please provide your email address?"
        })
    elif step == 2:
        new_state["customer_email"] = last_message
        new_state["contact_step"] = 3
        new_state["messages"].append({
            "role": "assistant",
            "content": "Thank you. Finally, could you please provide your phone number?"
        })
    elif step == 3:
        new_state["customer_phone"] = last_message
        new_state["contact_info_collected"] = True
        new_state["contact_step"] = 4
        success = contact_service.save_contact_info(
            new_state["customer_name"], new_state["customer_email"], new_state["customer_phone"]
        )
        response = (f"Thank you for providing your information. A customer service representative will contact you soon at "
                    f"{new_state['customer_email']} or {new_state['customer_phone']}. Is there anything else you'd like to add?")
        new_state["messages"].append({"role": "assistant", "content": response})
    return new_state

def continue_conversation(state: ChatbotState) -> ChatbotState:
    """Generates a response using the LLM service (Gemini API integration)."""
    new_state = state.copy()
    response_text = llm_service.generate_response(new_state["messages"], conversation_memory)
    new_state["messages"].append({"role": "assistant", "content": response_text})
    return new_state

def connect_to_agent(state: ChatbotState) -> ChatbotState:
    """Immediately transfer the conversation to a human representative."""
    new_state = state.copy()
    new_state["messages"].append({
        "role": "assistant",
        "content": "I'm connecting you to a human representative now. Please hold while I transfer your chat."
    })
    new_state["needs_human_agent"] = True
    new_state["contact_info_collected"] = True
    return new_state

def router(state: ChatbotState) -> str:
    """Determines the next node based on the last message."""
    last_msg = state["messages"][-1]["content"].lower()
    human_words = ["speak to a human", "talk to a human", "human representative",
                   "real person", "speak to an agent", "talk to a representative",
                   "connect me with a human", "human agent please", "agent", "representative"]
    if any(word in last_msg for word in human_words) and not state.get("needs_human_agent"):
        state["needs_human_agent"] = True
        state["contact_step"] = 0
        return "collect_contact_info"
    if state.get("needs_human_agent") and not state.get("contact_info_collected"):
        return "collect_contact_info"
    if any(term in last_msg for term in ["bye", "thank you", "thanks"]):
        return END
    if not state.get("order_lookup_attempted"):
        order_related = ("order" in last_msg and ("status" in last_msg or "track" in last_msg))
        has_id = bool(re.search(r"\b([a-f0-9]{32})\b", last_msg))
        if order_related or has_id:
            return "lookup_order"
    return "continue_conversation"

def create_chatbot():
    """Compiles the chatbot workflow using a state graph."""
    try:
        verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
        workflow = StateGraph(ChatbotState)

        def enhanced_continue_conversation(state: ChatbotState) -> ChatbotState:
            try:
                new_state = state.copy()
                last_msg = new_state["messages"][-1]["content"].lower()
                conversation_memory.add_message("user", last_msg)
                if state.get("current_order_id"):
                    conversation_memory.update_key_details("current_order_id", state["current_order_id"])
                if state.get("customer_name"):
                    conversation_memory.update_key_details("customer_name", state["customer_name"])
                response_text = llm_service.generate_response(new_state["messages"], conversation_memory)
                conversation_memory.add_message("assistant", response_text)
                new_state["messages"].append({"role": "assistant", "content": response_text})
                return new_state
            except Exception as e:
                logger.error(f"Error in enhanced response generation: {e}")
                raise

        workflow.add_node("continue_conversation", enhanced_continue_conversation)
        workflow.add_node("lookup_order", lookup_order)
        workflow.add_node("collect_contact_info", collect_contact_info)
        workflow.add_conditional_edges("continue_conversation", router)
        workflow.add_conditional_edges("lookup_order", router)
        workflow.add_conditional_edges("collect_contact_info", router)
        workflow.set_entry_point("continue_conversation")
        return workflow.compile()
    except Exception as e:
        logger.error(f"Chatbot creation failed: {e}")
        raise

def detect_intent(message: str) -> Optional[str]:
    lower_msg = message.lower()
    for intent, patterns in FAQ_CONFIG["intent_patterns"].items():
        if any(pattern in lower_msg for pattern in patterns):
            return intent
    human_keywords = ["speak to a human", "talk to a human", "human representative",
                      "real person", "speak to an agent", "talk to a representative",
                      "connect me with a human", "human agent please", "agent", "representative"]
    if any(word in lower_msg for word in human_keywords):
        return "human_agent"
    if ("order status" in lower_msg) or ("order" in lower_msg and "status" in lower_msg):
        return "order_status"
    return None

def chat_with_user(user_input: str, chat_state: Optional[Dict] = None) -> Dict:
    """Updates the chatbot state based on a user input."""
    try:
        chat_state = chat_state or reset_state()
        if not chat_state["messages"] or chat_state["messages"][-1]["content"] != user_input:
            chat_state["messages"].append({"role": "user", "content": user_input})
        intent = detect_intent(user_input)
        if intent and intent in FAQ_RESPONSES and intent not in ["greeting", "goodbye", "human_agent", "order_status"]:
            chat_state["messages"].append({"role": "assistant", "content": FAQ_RESPONSES[intent]})
            return chat_state
        if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
            cancel_terms = ["cancel", "nevermind", "never mind", "stop", "go back", "different question"]
            id_terms = ["customer id", "my id", "delivery status", "order status", "track"]
            if any(term in user_input.lower() for term in cancel_terms) or any(term in user_input.lower() for term in id_terms):
                chat_state["needs_human_agent"] = False
                chat_state["contact_step"] = 0
                if any(term in user_input.lower() for term in id_terms):
                    chat_state["messages"].append({
                        "role": "assistant",
                        "content": ("I understand you'd like to check your order status using your customer ID. "
                                    "Please provide your customer ID (a 32-character alphanumeric code).")
                    })
                else:
                    chat_state["messages"].append({
                        "role": "assistant",
                        "content": "I've canceled the request to speak with a human. How else may I help you?"
                    })
                return chat_state
        if intent == "human_agent" and not chat_state.get("needs_human_agent"):
            chat_state["needs_human_agent"] = True
            chat_state["contact_step"] = 1
            chat_state["messages"].append({
                "role": "assistant",
                "content": "I'll connect you with a human representative. Could you please provide your name?"
            })
            return chat_state
        if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
            return collect_contact_info(chat_state)
        if not chat_state.get("order_lookup_attempted") and re.search(r'\b([a-f0-9]{32})\b', user_input):
            return lookup_order(chat_state)
        if intent == "order_status" and not chat_state.get("order_lookup_attempted"):
            chat_state["messages"].append({
                "role": "assistant",
                "content": ("To check your order status, I'll need your order ID or customer ID. "
                            "The order ID is a 32-character code from your confirmation email. "
                            "Please provide your order ID.")
            })
            return chat_state
        try:
            chatbot = create_chatbot()
            result_state = chatbot.invoke(chat_state, config={"recursion_limit": 30})
            return result_state
        except Exception as e:
            logger.error(f"Chatbot processing error: {e}")
            chat_state["messages"].append({
                "role": "assistant",
                "content": ("I'm sorry, I don't have specific information about that. "
                            "Would you like to ask about return policy, shipping options, order status, or speak with a human representative?")
            })
            return chat_state
    except EnvironmentError as ee:
        logger.error(f"Credential error: {ee}")
        chat_state = chat_state or reset_state()
        chat_state["messages"].append({
            "role": "assistant",
            "content": "I'm having trouble accessing my services right now. Please try again later or contact support."
        })
        return chat_state
    except Exception as ex:
        logger.error(f"General error in chat_with_user: {ex}")
        chat_state = chat_state or reset_state()
        chat_state["messages"].append({
            "role": "assistant",
            "content": "I apologize for the trouble. Would you like to try asking your question again?"
        })
        return chat_state

if __name__ == "__main__":
    try:
        verify_credentials(["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS"])
        print("E-commerce Support Chatbot (type 'exit' to quit)")
        state = reset_state()
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() == "exit":
                break
            state = chat_with_user(user_input, state)
            print(f"\nAssistant: {state['messages'][-1]['content']}")
            if state.get("needs_human_agent") and state.get("contact_info_collected"):
                print("\nConversation has been handed off to a human representative.")
                break
    except Exception as e:
        logger.error(f"Chatbot startup failed: {e}")
        print("Error: Could not start chatbot. Please check credentials and try again.")
