# Description: E-commerce support chatbot.py using a state graph for conversation management.
import os
import sys
import time
import json
import re
import warnings
import numpy as np
import pandas as pd
import chromadb
import google.generativeai as genai
from functools import lru_cache
from typing import Dict, List, Any, Optional, TypedDict
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Filter warnings
warnings.filterwarnings("ignore", category=UserWarning, module="langchain_google_genai")

# Load environment variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Import local modules
from src.config import SYSTEM_PROMPT, ORDER_STATUS_DESCRIPTIONS, API_CONFIG
from src.utils import load_order_data, get_order_status, format_order_details, create_order_index, initialize_vector_db
from src.vector_db import (
    query_vector_db,
    get_order_by_id,
    get_orders_by_customer_id,
    cached_get_order_by_id,
    cached_get_orders_by_customer_id
)

class GeminiEmbeddingFunction:
    """Custom embedding function using Gemini API"""
    def __call__(self, input_texts):
        """Generate embeddings for a list of texts"""
        if not input_texts:
            return []

        # Clean and prepare texts
        cleaned_texts = [text.strip() for text in input_texts if text and text.strip()]
        if not cleaned_texts:
            return []

        try:
            # Use Gemini embedding model
            model = "models/embedding-001"
            embeddings = []

            # Process texts in batches to avoid API limits
            batch_size = 10
            for i in range(0, len(cleaned_texts), batch_size):
                batch = cleaned_texts[i:i+batch_size]
                batch_embeddings = [
                    genai.embed_content(model=model, content=text)["embedding"]
                    for text in batch
                ]
                embeddings.extend(batch_embeddings)

            return embeddings
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            # Return zero embeddings as fallback
            return [np.zeros(768) for _ in cleaned_texts]

def create_order_vector_db(orders_df: pd.DataFrame, db_path: str = "./data/vector_db") -> chromadb.Collection:
    """
    Create a vector database from order data.

    Args:
        orders_df: DataFrame containing order data
        db_path: Path to store the vector database

    Returns:
        ChromaDB collection with order data
    """
    # Ensure directory exists
    os.makedirs(db_path, exist_ok=True)

    # Initialize ChromaDB client
    client = chromadb.PersistentClient(path=db_path)

    # Create or get collection
    try:
        collection = client.get_collection(
            name="orders",
            embedding_function=GeminiEmbeddingFunction()
        )
        print("Using existing vector database")
    except ValueError:
        # Collection doesn't exist, create it
        collection = client.create_collection(
            name="orders",
            embedding_function=GeminiEmbeddingFunction()
        )
        print("Creating new vector database")

        # Prepare data for insertion
        documents = []
        metadatas = []
        ids = []

        for idx, row in orders_df.iterrows():
            # Create a text representation of the order for embedding
            order_text = f"""
            Order ID: {row['order_id']}
            Customer ID: {row.get('customer_id', 'Unknown')}
            Status: {row['order_status']}
            Purchase Date: {row['order_purchase_timestamp']}
            """

            # Add optional fields if available
            if pd.notna(row.get('order_approved_at')):
                order_text += f"Approved Date: {row['order_approved_at']}\n"
            if pd.notna(row.get('order_delivered_carrier_date')):
                order_text += f"Carrier Delivery Date: {row['order_delivered_carrier_date']}\n"
            if pd.notna(row.get('order_delivered_customer_date')):
                order_text += f"Customer Delivery Date: {row['order_delivered_customer_date']}\n"
            if pd.notna(row.get('order_estimated_delivery_date')):
                order_text += f"Estimated Delivery Date: {row['order_estimated_delivery_date']}\n"

            # Store the document, metadata, and ID
            documents.append(order_text)
            metadatas.append({
                "order_id": row['order_id'],
                "customer_id": str(row.get('customer_id', '')),
                "status": row['order_status'],
                "purchase_date": row['order_purchase_timestamp']
            })
            ids.append(str(idx))

            # Add in batches to avoid memory issues
            if len(documents) >= 1000:
                collection.add(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents = []
                metadatas = []
                ids = []

        # Add any remaining documents
        if documents:
            collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

    return collection

def query_vector_db(query: str, collection: chromadb.Collection, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Query the vector database for relevant orders.

    Args:
        query: User query text
        collection: ChromaDB collection to query
        top_k: Number of results to return

    Returns:
        List of relevant order data
    """
    try:
        # Query the collection
        results = collection.query(
            query_texts=[query],
            n_results=top_k
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })

        return formatted_results
    except Exception as e:
        print(f"Error querying vector database: {e}")
        return []

def get_order_by_id(order_id: str, collection: chromadb.Collection) -> Dict[str, Any]:
    """
    Get order details by order ID.

    Args:
        order_id: Order ID to look up
        collection: ChromaDB collection to query

    Returns:
        Order details or None if not found
    """
    try:
        # Query by metadata
        results = collection.query(
            query_texts=[""],
            where={"order_id": order_id},
            n_results=1
        )

        if results["ids"][0]:
            return {
                "id": results["ids"][0][0],
                "metadata": results["metadatas"][0][0],
                "document": results["documents"][0][0]
            }
        return None
    except Exception as e:
        print(f"Error getting order by ID: {e}")
        return None

def get_orders_by_customer_id(customer_id: str, collection: chromadb.Collection) -> List[Dict[str, Any]]:
    """
    Get all orders for a customer.

    Args:
        customer_id: Customer ID to look up
        collection: ChromaDB collection to query

    Returns:
        List of order details
    """
    try:
        # Query by metadata
        results = collection.query(
            query_texts=[""],
            where={"customer_id": customer_id},
            n_results=100  # Set a high limit to get all orders
        )

        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append({
                "id": results["ids"][0][i],
                "metadata": results["metadatas"][0][i],
                "document": results["documents"][0][i]
            })

        return formatted_results
    except Exception as e:
        print(f"Error getting orders by customer ID: {e}")
        return []
from src.vector_db import query_vector_db, get_order_by_id, get_orders_by_customer_id
from functools import lru_cache

# Cached version of load_order_data
@lru_cache(maxsize=1)
def load_order_data_cached():
    """Load order data with caching for better performance."""
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
    contact_step: int  # Track which step of contact collection we're on

def create_chatbot():
    """Initialize and return the e-commerce support chatbot graph."""
    # Try Gemini first, fall back to OpenAI if needed
    try:
        # Initialize LLM with Gemini
        llm = ChatGoogleGenerativeAI(
            model=API_CONFIG["gemini"]["model"],
            temperature=API_CONFIG["gemini"]["temperature"],
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            convert_system_message_to_human=True,
            timeout=API_CONFIG["gemini"]["timeout_seconds"]
        )
        print("Using Gemini API")
    except Exception as e:
        print(f"Gemini API error: {e}. Falling back to OpenAI.")
        # Fall back to OpenAI
        llm = ChatOpenAI(
            model=API_CONFIG["openai"]["model"],
            temperature=API_CONFIG["openai"]["temperature"],
            api_key=os.getenv("OPENAI_API_KEY"),
            max_tokens=API_CONFIG["openai"]["max_tokens"],
            request_timeout=API_CONFIG["openai"]["timeout_seconds"]
        )

    # Load order data
    orders_df = load_order_data_cached()

    # Initialize vector database
    vector_collection = initialize_vector_db(orders_df)

    # Create indexes for faster lookups
    order_index, customer_index = create_order_index(orders_df)

    # Define nodes
    
    def lookup_order(state: ChatbotState) -> ChatbotState:
        """Extract order ID or customer ID and look up status."""
        start_time = time.time()
        last_message = state["messages"][-1]["content"]

        # Simple regex pattern to extract order ID or customer ID
        id_match = re.search(r'\b([a-f0-9]{32})\b', last_message)

        if id_match:
            extracted_id = id_match.group(1)
            # Try direct dictionary lookup first (fastest)
            if extracted_id in order_index:
                order_data = order_index[extracted_id]
                status = order_data['order_status']
                details = {
                    'purchase_date': order_data['order_purchase_timestamp'],
                    'delivery_date': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None,
                    'approved_date': order_data['order_approved_at'] if pd.notna(order_data['order_approved_at']) else None,
                    'estimated_delivery': order_data['order_estimated_delivery_date'] if pd.notna(order_data['order_estimated_delivery_date']) else None,
                    'actual_delivery': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None
                }
                response = format_order_details(extracted_id, status, details)
            elif extracted_id in customer_index:
                # Use direct dictionary access for customer lookup
                customer_orders = customer_index[extracted_id]
                if len(customer_orders) == 1:
                    order_data = customer_orders[0]
                    order_id = order_data['order_id']
                    status = order_data['order_status']
                    details = {
                        'purchase_date': order_data['order_purchase_timestamp'],
                        'delivery_date': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None,
                        'approved_date': order_data['order_approved_at'] if pd.notna(order_data['order_approved_at']) else None,
                        'estimated_delivery': order_data['order_estimated_delivery_date'] if pd.notna(order_data['order_estimated_delivery_date']) else None,
                        'actual_delivery': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None
                    }
                    response = f"I found an order for your customer ID. {format_order_details(order_id, status, details)}"
                else:
                    sorted_orders = sorted(customer_orders,
                                        key=lambda x: pd.to_datetime(x['order_purchase_timestamp']),
                                        reverse=True)
                    recent_order = sorted_orders[0]
                    order_id = recent_order['order_id']
                    status = recent_order['order_status']
                    details = {
                        'purchase_date': recent_order['order_purchase_timestamp'],
                        'delivery_date': recent_order['order_delivered_customer_date'] if pd.notna(recent_order['order_delivered_customer_date']) else None,
                        'approved_date': recent_order['order_approved_at'] if pd.notna(recent_order['order_approved_at']) else None,
                        'estimated_delivery': recent_order['order_estimated_delivery_date'] if pd.notna(recent_order['order_estimated_delivery_date']) else None,
                        'actual_delivery': recent_order['order_delivered_customer_date'] if pd.notna(recent_order['order_delivered_customer_date']) else None
                    }
                    response = f"I found {len(customer_orders)} orders for your customer ID. Here's the status of your most recent order: {format_order_details(order_id, status, details)}"
            else:
                # Fall back to vector search using cached functions
                from src.vector_db import cached_get_order_by_id, cached_get_orders_by_customer_id

                # Try order ID first
                order_result = cached_get_order_by_id(extracted_id, vector_collection)
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
                    response = format_order_details(extracted_id, status, details)
                else:
                    # Try customer ID
                    customer_orders = cached_get_orders_by_customer_id(extracted_id, vector_collection)
                    if customer_orders:
                        if len(customer_orders) == 1:
                            metadata = customer_orders[0]["metadata"]
                            order_id = metadata["order_id"]
                            status = metadata["status"]
                            details = {
                                'purchase_date': metadata['purchase_date'],
                                'delivery_date': metadata.get('customer_delivery_date'),
                                'approved_date': metadata.get('approved_date'),
                                'estimated_delivery': metadata.get('estimated_delivery_date'),
                                'actual_delivery': metadata.get('customer_delivery_date')
                            }
                            response = f"I found an order for your customer ID. {format_order_details(order_id, status, details)}"
                        else:
                            response = f"I found {len(customer_orders)} orders for your customer ID. Here's the most recent one: "
                            sorted_orders = sorted(customer_orders,
                                                key=lambda x: pd.to_datetime(x["metadata"]['purchase_date']),
                                                reverse=True)
                            metadata = sorted_orders[0]["metadata"]
                            order_id = metadata["order_id"]
                            status = metadata["status"]
                            details = {
                                'purchase_date': metadata['purchase_date'],
                                'delivery_date': metadata.get('customer_delivery_date'),
                                'approved_date': metadata.get('approved_date'),
                                'estimated_delivery': metadata.get('estimated_delivery_date'),
                                'actual_delivery': metadata.get('customer_delivery_date')
                            }
                            response += format_order_details(order_id, status, details)
                    else:
                        response = f"I couldn't find any orders with ID {extracted_id}. Please check the ID and try again."
        else:
            # No ID found in message
            response = "I'd be happy to help check your order status. Could you please provide your order ID or customer ID? It should be a 32-character alphanumeric code."

        print(f"Lookup completed in {time.time() - start_time:.4f} seconds")

        new_state = state.copy()
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
            import csv
            import os

            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)

            csv_path = os.path.join("data", "contact_requests.csv")
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, 'a', newline='') as csvfile:
                fieldnames = ['name', 'email', 'phone']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                writer.writerow({
                    'name': new_state["customer_name"],
                    'email': new_state["customer_email"],
                    'phone': new_state["customer_phone"]
                })

            new_state["messages"].append({
                "role": "assistant",
                "content": f"Thank you for providing your information. A customer service representative will contact you soon at {new_state['customer_email']} or {new_state['customer_phone']}. Is there anything else you'd like to add before I submit your request?"
            })

        return new_state

    def continue_conversation(state: ChatbotState) -> ChatbotState:
        """Process the user message and generate a response."""
        messages = state["messages"]
        last_message = messages[-1]["content"].lower()
        new_state = state.copy()

        # Check if user wants to speak to a human - be more specific with the pattern matching
        human_keywords = ["speak to a human", "talk to a human", "human representative",
                        "real person", "speak to an agent", "talk to a representative",
                        "connect me with a human", "human agent please"]

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

        # Format conversation history for the LLM
        prompt_messages = [
            SystemMessage(content=SYSTEM_PROMPT or "You are a helpful e-commerce support assistant."),
        ]

        # Ensure no empty messages are sent to the API
        for message in messages:
            if message["role"] == "user" and message["content"].strip():
                prompt_messages.append(HumanMessage(content=message["content"]))
            elif message["role"] == "assistant" and message["content"].strip():
                prompt_messages.append(AIMessage(content=message["content"]))

        # Generate response with better error handling
        try:
            response = llm.invoke(prompt_messages)
            response_text = response.content
        except Exception as e:
            print(f"Error generating response: {e}")
            # More robust fallback
            try:
                # Try with a simpler prompt if the full conversation fails
                simple_prompt = [
                    SystemMessage(content="You are a helpful e-commerce support assistant."),
                    HumanMessage(content=messages[-1]["content"])
                ]
                response = llm.invoke(simple_prompt)
                response_text = response.content
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
                response_text = "I'm having trouble processing your request. Could you please try again?"

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

    # Define the router function
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

    # Build the graph
    workflow = StateGraph(ChatbotState)

    # Add nodes
    workflow.add_node("lookup_order", lookup_order)
    workflow.add_node("connect_to_agent", connect_to_agent)
    workflow.add_node("collect_contact_info", collect_contact_info)
    workflow.add_node("continue_conversation", continue_conversation)

    # Add conditional edges with the router function
    workflow.add_conditional_edges("lookup_order", router)
    workflow.add_conditional_edges("collect_contact_info", router)
    workflow.add_conditional_edges("continue_conversation", router)

    # Set the entry point
    workflow.set_entry_point("continue_conversation")

    return workflow.compile()

def chat_with_user(user_input: str, chat_state: Optional[Dict] = None) -> Dict:
    """Process a single user message and return updated state."""
    import re

    if chat_state is None:
        chat_state = {
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

    # Add user message to state
    chat_state["messages"].append({"role": "user", "content": user_input})

    # Check if user wants to cancel human agent request
    user_input_lower = user_input.lower()
    if chat_state.get("needs_human_agent") and not chat_state.get("contact_info_collected"):
        cancel_keywords = ["cancel", "nevermind", "never mind", "stop", "go back", "different question"]
        customer_id_keywords = ["customer id", "my id", "delivery status", "order status", "track"]

        if any(keyword in user_input_lower for keyword in cancel_keywords) or any(keyword in user_input_lower for keyword in customer_id_keywords):
            # Reset the human agent flow
            chat_state["needs_human_agent"] = False
            chat_state["contact_step"] = 0

            # If they're asking about customer ID specifically
            if any(keyword in user_input_lower for keyword in customer_id_keywords):
                response = "I understand you'd like to check your order status using your customer ID. Could you please provide your customer ID? It should be a 32-character alphanumeric code."
                chat_state["messages"].append({"role": "assistant", "content": response})
                return chat_state
            else:
                response = "I've canceled the request to speak with a human representative. How else can I help you today?"
                chat_state["messages"].append({"role": "assistant", "content": response})
                return chat_state

    # Return policy FAQ
    if any(phrase in user_input_lower for phrase in ["return policy", "policy on return", "can i return",
                                                    "how to return", "policy for returns", "returned items"]):
        response = """Our return policy is as follows:

1. Items can be returned within 30 days of delivery for a full refund.
2. Products must be in original packaging and unused condition.
3. For electronics, returns are accepted within 15 days and must include all accessories.
4. Shipping costs for returns are covered by the customer unless the item was defective.
5. Refunds are processed within 5-7 business days after we receive the returned item.

Would you like more information about a specific aspect of our return policy?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Shipping policy FAQ
    if any(phrase in user_input_lower for phrase in ["shipping policy", "delivery policy", "shipping time",
                                                     "how long shipping", "shipping cost"]):
        response = """Our shipping policy:

1. Standard shipping (5-7 business days): Free for orders over $35, otherwise $4.99
2. Express shipping (2-3 business days): $9.99
3. Next-day delivery (where available): $19.99
4. International shipping available to select countries

Delivery times may vary based on your location and product availability. You can track your shipment using the order ID provided in your confirmation email.

Do you have any other questions about shipping?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Payment methods FAQ
    if any(phrase in user_input_lower for phrase in ["payment method", "payment option", "how to pay",
                                                    "accept payment", "credit card", "debit card", "paypal"]):
        response = """We accept the following payment methods:

1. Credit cards (Visa, Mastercard, American Express, Discover)
2. Debit cards
3. PayPal
4. Store credit/gift cards
5. Apple Pay and Google Pay (on mobile)

All payment information is securely processed and encrypted. We do not store your full credit card details on our servers.

Is there anything specific about our payment options you'd like to know?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Warranty information FAQ
    if any(phrase in user_input_lower for phrase in ["warranty", "guarantee", "product warranty",
                                                    "warranty policy", "warranty coverage"]):
        response = """Our warranty policy:

1. Most products come with a standard 1-year manufacturer's warranty.
2. Electronics typically include a 90-day warranty against defects.
3. Extended warranties are available for purchase on select items.
4. Warranty claims require proof of purchase and the original packaging if possible.
5. Warranties cover manufacturing defects but not damage from misuse or accidents.

To make a warranty claim, please contact our customer service with your order details and a description of the issue.

Do you need help with a specific warranty claim?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Order cancellation FAQ
    if any(phrase in user_input_lower for phrase in ["cancel order", "cancellation policy", "how to cancel",
                                                    "cancel my purchase", "stop my order"]):
        response = """Order cancellation information:

1. Orders can be cancelled within 1 hour of placement with no penalty.
2. Orders that haven't shipped can usually be cancelled through your account dashboard.
3. For orders that have already shipped, you'll need to wait for delivery and then follow the return process.
4. Cancellation requests are typically processed within 24 hours.
5. Refunds for cancelled orders are issued to the original payment method within 3-5 business days.

To cancel an order, please log into your account or provide your order ID.

Would you like to cancel a specific order?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Gift cards FAQ
    if any(phrase in user_input_lower for phrase in ["gift card", "gift certificate", "store credit",
                                                    "gift card balance", "redeem gift card"]):
        response = """Gift card information:

1. Gift cards are available in amounts from $10 to $500.
2. Digital gift cards are delivered via email within 24 hours of purchase.
3. Physical gift cards can be shipped to any address (standard shipping rates apply).
4. Gift cards do not expire and have no maintenance fees.
5. Lost or stolen gift cards cannot be replaced unless registered.

To check your gift card balance, please visit our website and enter your gift card number and PIN.

Can I help you purchase a gift card or check a balance?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Contact information FAQ
    if any(phrase in user_input_lower for phrase in ["contact info", "contact information", "phone number",
                                                    "email address", "contact us", "customer service contact"]):
        response = """Our contact information:

Customer Service Hours:
- Monday to Friday: 8:00 AM - 8:00 PM EST
- Saturday: 9:00 AM - 6:00 PM EST
- Sunday: 10:00 AM - 5:00 PM EST

Phone: +1-800-123-4567
Email: support@ecommerce-example.com
Live Chat: Available on our website during business hours

For the fastest response, please have your order number ready when contacting us.

Would you like me to connect you with a customer service representative?"""

        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # More precise detection of human agent requests - expanded keywords
    human_keywords = ["speak to a human", "talk to a human", "human representative",
                    "real person", "speak to an agent", "talk to a representative",
                    "connect me with a human", "human agent please", "agent", "representative",
                    "connect me to an agent", "need an agent", "talk to agent", "speak to agent"]

    # Check for specific phrases first
    needs_human = False
    for phrase in human_keywords:
        if phrase in user_input_lower:
            needs_human = True
            break

    # Also check for standalone keywords, but be more careful
    if not needs_human:
        standalone_words = ["human", "representative", "agent", "person"]
        # Only trigger if these words appear alone or as main request
        # Avoid triggering when asking about policies related to these words
        if any(word in user_input_lower.split() for word in standalone_words) and not any(policy_word in user_input_lower for policy_word in ["policy", "policies", "return", "shipping", "warranty"]):
            needs_human = True

    #Direct handling of human agent requests with improved detection
    if any(keyword in user_input_lower for keyword in human_keywords) and not chat_state.get("needs_human_agent"):
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
                    'name': chat_state["customer_name"],
                    'email': chat_state["customer_email"],
                    'phone': chat_state["customer_phone"]
                })

            chat_state["messages"].append({
                "role": "assistant",
                "content": f"Thank you for providing your information. A customer service representative will contact you soon at {chat_state['customer_email']} or {chat_state['customer_phone']}. Is there anything else you'd like to add before I submit your request?"
            })
            return chat_state

    # Direct handling of order lookups with IDs
    if not chat_state.get("order_lookup_attempted") and re.search(r'\b([a-f0-9]{32})\b', user_input):
        id_match = re.search(r'\b([a-f0-9]{32})\b', user_input)
        extracted_id = id_match.group(1)
        print(f"Attempting direct lookup for ID: {extracted_id}")  # Debug log

        try:
            orders_df = load_order_data_cached()
            order_index, customer_index = create_order_index(orders_df)

            if extracted_id in order_index:
                order_data = order_index[extracted_id]
                status = order_data['order_status']
                details = {
                    'purchase_date': order_data['order_purchase_timestamp'],
                    'delivery_date': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None,
                    'approved_date': order_data['order_approved_at'] if pd.notna(order_data['order_approved_at']) else None,
                    'estimated_delivery': order_data['order_estimated_delivery_date'] if pd.notna(order_data['order_estimated_delivery_date']) else None,
                    'actual_delivery': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None
                }
                response = format_order_details(extracted_id, status, details)
                print(f"Found order {extracted_id}: {status}")  # Debug log
                chat_state["messages"].append({"role": "assistant", "content": response})
                chat_state["order_lookup_attempted"] = True
                chat_state["current_order_id"] = extracted_id
                return chat_state
            elif extracted_id in customer_index:
                customer_orders = customer_index[extracted_id]
                order_count = len(customer_orders)
                print(f"ID {extracted_id} identified as customer ID with {order_count} orders")  # Debug log
                if order_count == 1:
                    order_data = customer_orders[0]
                    order_id = order_data['order_id']
                    status = order_data['order_status']
                    details = {
                        'purchase_date': order_data['order_purchase_timestamp'],
                        'delivery_date': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None,
                        'approved_date': order_data['order_approved_at'] if pd.notna(order_data['order_approved_at']) else None,
                        'estimated_delivery': order_data['order_estimated_delivery_date'] if pd.notna(order_data['order_estimated_delivery_date']) else None,
                        'actual_delivery': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None
                    }
                    response = f"I found an order for customer ID {extracted_id}. {format_order_details(order_id, status, details)}"
                else:
                    sorted_orders = sorted(customer_orders, key=lambda x: pd.to_datetime(x['order_purchase_timestamp']), reverse=True)
                    recent_order = sorted_orders[0]
                    order_id = recent_order['order_id']
                    status = recent_order['order_status']
                    details = {
                        'purchase_date': recent_order['order_purchase_timestamp'],
                        'delivery_date': recent_order['order_delivered_customer_date'] if pd.notna(recent_order['order_delivered_customer_date']) else None,
                        'approved_date': recent_order['order_approved_at'] if pd.notna(recent_order['order_approved_at']) else None,
                        'estimated_delivery': recent_order['order_estimated_delivery_date'] if pd.notna(recent_order['order_estimated_delivery_date']) else None,
                        'actual_delivery': recent_order['order_delivered_customer_date'] if pd.notna(recent_order['order_delivered_customer_date']) else None
                    }
                    response = f"I found {order_count} orders for customer ID {extracted_id}. Here's the most recent: {format_order_details(order_id, status, details)}"
                chat_state["messages"].append({"role": "assistant", "content": response})
                chat_state["order_lookup_attempted"] = True
                chat_state["current_order_id"] = order_id if order_count == 1 else None
                return chat_state
            else:
                print(f"ID {extracted_id} not found in order_index or customer_index")  # Debug log
                response = f"I couldn't find any orders or customers with ID {extracted_id}. Please check the ID and try again."
                chat_state["messages"].append({"role": "assistant", "content": response})
                chat_state["order_lookup_attempted"] = True
                return chat_state
        except Exception as e:
            print(f"Direct lookup failed: {e}")  # Debug log
            # Continue with graph-based processing

    # Customer ID lookup without explicit order ID pattern
    if not chat_state.get("order_lookup_attempted") and any(phrase in user_input_lower for phrase in ["my customer id", "customer id", "my id", "how many orders", "my orders"]):
        # Extract potential customer ID - look for 32-char hex pattern
        id_match = re.search(r'\b([a-f0-9]{32})\b', user_input)
        if id_match:
            customer_id = id_match.group(1)
            print(f"Attempting customer ID lookup for ID: {customer_id}")  # Debug log

            try:
                orders_df = load_order_data_cached()
                order_index, customer_index = create_order_index(orders_df)

                if customer_id in customer_index:
                    customer_orders = customer_index[customer_id]
                    order_count = len(customer_orders)
                    print(f"Found {order_count} orders for customer ID {customer_id}")  # Debug log

                    if order_count == 1:
                        order_data = customer_orders[0]
                        order_id = order_data['order_id']
                        status = order_data['order_status']
                        details = {
                            'purchase_date': order_data['order_purchase_timestamp'],
                            'delivery_date': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None,
                            'approved_date': order_data['order_approved_at'] if pd.notna(order_data['order_approved_at']) else None,
                            'estimated_delivery': order_data['order_estimated_delivery_date'] if pd.notna(order_data['order_estimated_delivery_date']) else None,
                            'actual_delivery': order_data['order_delivered_customer_date'] if pd.notna(order_data['order_delivered_customer_date']) else None
                        }
                        response = f"You have 1 order with customer ID {customer_id}. Here are the details: {format_order_details(order_id, status, details)}"
                    else:
                        response = f"You have {order_count} orders with customer ID {customer_id}. Here's a summary:\n\n"
                        sorted_orders = sorted(customer_orders, key=lambda x: pd.to_datetime(x['order_purchase_timestamp']), reverse=True)
                        for i, order in enumerate(sorted_orders[:5]):
                            response += f"{i+1}. Order ID: {order['order_id']}\n   Status: {order['order_status']}\n   Purchased: {order['order_purchase_timestamp']}\n"
                            if i < len(sorted_orders) - 1:
                                response += "\n"
                        if order_count > 5:
                            response += f"\nShowing 5 most recent orders out of {order_count} total."
                    chat_state["messages"].append({"role": "assistant", "content": response})
                    chat_state["order_lookup_attempted"] = True
                    return chat_state
                else:
                    print(f"Customer ID {customer_id} not found in customer_index")  # Debug log
                    response = f"I couldn't find any orders with customer ID {customer_id}. Please check the ID and try again."
                    chat_state["messages"].append({"role": "assistant", "content": response})
                    chat_state["order_lookup_attempted"] = True
                    return chat_state
            except Exception as e:
                print(f"Customer ID lookup failed: {e}")  # Debug log
                # Continue with graph-based processing

    # General order status question without specific ID
    if not chat_state.get("order_lookup_attempted") and "order status" in user_input_lower:
        response = """To check your order status, I'll need your order ID or customer ID.

The order ID is a 32-character code that was included in your order confirmation email. It looks something like: e481f51cbdc54678b7cc49136f2d6af7

Could you please provide your order ID?"""
        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Greeting/Hello
    if any(greeting in user_input_lower for greeting in ["hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening"]):
        response = "Hello! Welcome to our e-commerce support. How can I help you today? You can ask about order status, return policies, shipping information, or connect with a human representative."
        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Thank you/goodbye
    if any(phrase in user_input_lower for phrase in ["thank you", "thanks", "bye", "goodbye", "see you", "that's all"]):
        response = "You're welcome! Thank you for contacting our support. If you have any other questions in the future, don't hesitate to reach out. Have a great day!"
        chat_state["messages"].append({"role": "assistant", "content": response})
        return chat_state

    # Get chatbot
    chatbot = create_chatbot()

    # Process message through graph with higher recursion limit
    try:
        result = chatbot.invoke(chat_state, config={"recursion_limit": 30})
        return result
    except Exception as e:
        print(f"Error in chatbot processing: {e}")

        # Fallback response if graph fails
        chat_state["messages"].append({
            "role": "assistant",
            "content": "I'm sorry, I don't have specific information about that. Would you like to ask about our return policy, shipping options, order status, or speak with a human representative?"
        })
        return chat_state
    
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