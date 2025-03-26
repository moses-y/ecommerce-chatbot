# src/utils.py
# src/utils.py

import os
import sys
import pandas as pd
import json
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from functools import lru_cache
import logging # Added logging

# Configure logging for utils
logger = logging.getLogger(__name__)
# Basic config if not already set by chatbot.py (adjust level as needed)
if not logger.hasHandlers():
     logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


# Add the project root to the Python path
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Ensure path is correct

# Import configuration - ensure this path is correct
try:
    from src.config import ORDER_STATUS_DESCRIPTIONS
except ImportError:
    logger.error("Could not import ORDER_STATUS_DESCRIPTIONS from src.config. Using default descriptions.")
    # Provide fallback descriptions if import fails
    ORDER_STATUS_DESCRIPTIONS = {
        'created': 'Your order has been created in the system.',
        'approved': 'Your payment has been approved.',
        'processing': 'Your order is being processed and prepared for shipment.',
        'invoiced': 'Your order has been invoiced.',
        'shipped': 'Your order has been shipped and is on its way.',
        'delivered': 'Your order has been delivered.',
        'canceled': 'Your order has been canceled.',
        'unavailable': 'Some items in your order are currently unavailable.'
        # Add other statuses from your dataset
    }


# --- Keep load_order_data, create_order_index, get_order_status (though get_order_status is now less used by chatbot.py) ---
@lru_cache(maxsize=1)
def load_order_data(use_cache: bool = True) -> pd.DataFrame:
    """Loads order data, handling potential errors and caching."""
    data_dir = "data"
    cache_path = os.path.join(data_dir, "cached_orders.csv")
    original_path = os.path.join(data_dir, "olist_orders_dataset.csv") # Ensure this filename is correct

    os.makedirs(data_dir, exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached order data from {cache_path}")
            return pd.read_csv(cache_path)
        except Exception as e:
            logger.warning(f"Error loading cached data from {cache_path}: {e}. Trying original file.")

    if os.path.exists(original_path):
        try:
            logger.info(f"Loading original order data from {original_path}")
            orders_df = pd.read_csv(original_path)
            # Attempt to cache the loaded data
            try:
                orders_df.to_csv(cache_path, index=False)
                logger.info(f"Cached order data to {cache_path}")
            except Exception as cache_e:
                logger.error(f"Failed to cache order data to {cache_path}: {cache_e}")
            return orders_df
        except Exception as load_e:
            logger.error(f"Error loading original order data from {original_path}: {load_e}", exc_info=True)
            # Return empty DataFrame or raise error depending on desired behavior
            return pd.DataFrame() # Return empty DataFrame on error
            # raise RuntimeError(f"Could not load order data from {original_path}") from load_e
    else:
        logger.error(f"Order dataset not found at {original_path}. Returning empty DataFrame.")
        return pd.DataFrame() # Return empty DataFrame if file not found
        # raise FileNotFoundError(f"Order dataset not found at {original_path}")

def create_order_index(orders_df: pd.DataFrame) -> Tuple[Dict[str, Dict], Dict[str, List[Dict]]]:
    """Create indexed dictionaries for faster order lookups."""
    order_index = {}
    customer_index = {}

    if orders_df.empty:
        logger.warning("Cannot create order index from empty DataFrame.")
        return order_index, customer_index

    logger.info(f"Creating order index from DataFrame with {len(orders_df)} rows.")
    for _, row in orders_df.iterrows():
        try:
            # Ensure required keys exist
            order_id = row['order_id']
            customer_id = row.get('customer_id') # Use .get() for optional keys
            order_data = row.to_dict()

            # Index by order ID
            order_index[order_id] = order_data

            # Index by customer ID (if present)
            if customer_id and pd.notna(customer_id):
                if customer_id not in customer_index:
                    customer_index[customer_id] = []
                customer_index[customer_id].append(order_data)
        except KeyError as ke:
            logger.warning(f"Skipping row due to missing key: {ke} in row data: {row.to_dict()}")
        except Exception as e:
             logger.warning(f"Skipping row due to unexpected error during indexing: {e} in row data: {row.to_dict()}")


    logger.info(f"Order index created. Order IDs: {len(order_index)}, Customer IDs: {len(customer_index)}")
    return order_index, customer_index

# get_order_status might still be useful for other purposes, but chatbot.py's OrderService now handles lookup directly.
def get_order_status(orders_df: pd.DataFrame, order_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Get the status and details of an order (legacy function, formatting moved).
    Note: Formatting logic is now primarily in format_order_details.
    """
    if orders_df.empty:
        logger.warning("get_order_status called with empty DataFrame.")
        return None, None

    order = orders_df[orders_df['order_id'] == order_id]

    if order.empty:
        logger.info(f"Order ID {order_id} not found in provided DataFrame for get_order_status.")
        return None, None

    order_row = order.iloc[0]
    status = order_row.get('order_status')

    # Return raw details, formatting happens in format_order_details
    details = {
        'purchase_date': order_row.get('order_purchase_timestamp'),
        'approved_date': order_row.get('order_approved_at'),
        'estimated_delivery': order_row.get('order_estimated_delivery_date'),
        'actual_delivery': order_row.get('order_delivered_customer_date')
        # Add any other relevant raw fields needed by format_order_details
    }
    logger.debug(f"get_order_status found raw details for {order_id}: Status={status}, Details={details}")
    return status, details


# --- MODIFIED: format_order_details ---
def format_order_details(order_id: str, status: Optional[str], details: Optional[Dict[str, Any]]) -> str:
    """
    Format order details into a human-readable response.
    Handles raw date values from the details dictionary.

    Args:
        order_id: ID of the order.
        status: Current status of the order (e.g., 'shipped', 'delivered').
        details: Dictionary containing raw order details like timestamps.
                 Expected keys: 'purchase_date', 'approved_date', 'estimated_delivery', 'actual_delivery'.

    Returns:
        Formatted string with order information.
    """
    # --- Input Validation ---
    if not order_id:
        logger.warning("format_order_details called with no order_id.")
        return "I need an order ID to provide details."
    if not status:
        logger.warning(f"format_order_details called for order {order_id} with no status.")
        status = "unknown" # Default status if None
    if not details:
        logger.warning(f"format_order_details called for order {order_id} with no details dictionary.")
        details = {} # Use empty dict if None

    logger.debug(f"Formatting details for Order ID: {order_id}, Status: {status}, Raw Details: {details}")

    # --- Helper for Date Formatting ---
    def format_date(date_value: Any, default: str = "Not available") -> str:
        """Safely formats a date value."""
        if pd.isna(date_value):
            return default
        try:
            # Convert to datetime object first
            dt_obj = pd.to_datetime(date_value)
            # Format as Month Day, Year
            return dt_obj.strftime('%B %d, %Y')
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not format date value '{date_value}': {e}")
            return default # Return default if formatting fails

    # --- Format Dates from Details ---
    purchase_date_str = format_date(details.get('purchase_date'))
    approved_date_str = format_date(details.get('approved_date'))
    actual_delivery_str = format_date(details.get('actual_delivery'), default="Not yet delivered")

    # --- Handle Estimated Delivery and Overdue Status ---
    estimated_delivery_str = "Not available"
    is_overdue = False
    est_delivery_val = details.get('estimated_delivery')
    if pd.notna(est_delivery_val):
        try:
            est_delivery_date = pd.to_datetime(est_delivery_val)
            estimated_delivery_str = est_delivery_date.strftime('%B %d, %Y')
            # Check if delivery is overdue (only if not delivered or canceled)
            if est_delivery_date < datetime.now() and status not in ['delivered', 'canceled']:
                estimated_delivery_str += " (Potentially Overdue)"
                is_overdue = True
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not format estimated delivery date value '{est_delivery_val}': {e}")
            estimated_delivery_str = "Not available" # Fallback

    # --- Get Status Description ---
    status_description = ORDER_STATUS_DESCRIPTIONS.get(status, f"Unknown status ('{status}')")

    # --- Construct the Response ---
    # Start with basic status and description
    response_lines = [
        f"Okay, here's the information for order #{order_id}:",
        f"Status: **{status.capitalize()}**",
        f"   - {status_description}",
        f"Purchased on: {purchase_date_str}"
    ]

    # Add relevant dates based on status
    if status not in ['canceled', 'created']: # Don't show approval/delivery for canceled/created
        if approved_date_str != "Not available":
            response_lines.append(f"Payment Approved on: {approved_date_str}")

        if estimated_delivery_str != "Not available":
            response_lines.append(f"Estimated Delivery: {estimated_delivery_str}")

        if status == 'delivered':
            response_lines.append(f"Delivered on: {actual_delivery_str}")
        elif status == 'shipped':
             # --- Tracking Link Placeholder ---
             # Add logic here to get the actual tracking link if available
             # This might involve looking up carrier info + tracking number in 'details'
             # For now, using a placeholder.
             tracking_link_placeholder = "[Tracking information not available in this demo]"
             # Example if you had tracking_number and carrier in details:
             # tracking_number = details.get("tracking_number")
             # carrier = details.get("carrier")
             # if tracking_number and carrier:
             #    tracking_link_placeholder = f"Track via {carrier}: {tracking_number} [Link placeholder]"

             response_lines.append(f"Tracking: {tracking_link_placeholder}")

    # Add additional context/next steps based on status
    if status == 'processing':
        response_lines.append("\nYour order is being prepared. You'll receive an update once it ships.")
    elif status == 'shipped':
        response_lines.append("\nYour order is on its way!")
        if is_overdue:
             response_lines.append("It seems to be past the estimated delivery date. Please allow a little extra time, or contact support if it doesn't arrive soon.")
    elif status == 'canceled':
        response_lines.append("\nThis order was canceled. If this seems incorrect, please contact customer support.")
    elif status == 'delivered':
        if is_overdue: # Check if it was delivered late
             response_lines.append("\nWe apologize if the delivery was later than estimated.")
        response_lines.append("\nWe hope you enjoy your items!")
    elif is_overdue: # For statuses other than shipped/delivered where it's overdue
         response_lines.append("\nThis order is past its estimated delivery date. Please contact customer support for assistance.")


    # Join lines into a single response string
    response = "\n".join(response_lines)
    logger.debug(f"Formatted response for {order_id}: {response}")
    return response


# --- Keep load_return_policies and initialize_vector_db ---
def load_return_policies(use_cache: bool = True) -> Dict[str, Any]:
    """Loads return policies, handling potential errors and caching."""
    data_dir = "data"
    cache_path = os.path.join(data_dir, "cached_policies.json")
    # Define default policies structure here or load from a default config file
    default_policies = {
        "standard": {"days": 30, "condition": "unused", "notes": "Must be in original packaging."},
        # Add other default policy categories
    }

    os.makedirs(data_dir, exist_ok=True)

    if use_cache and os.path.exists(cache_path):
        try:
            logger.info(f"Loading cached policies from {cache_path}")
            with open(cache_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Error loading cached policies from {cache_path}: {e}. Using default policies.")
            return default_policies

    # If no cache or cache failed, use default and try to save cache
    logger.info("Using default return policies and attempting to cache.")
    try:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(default_policies, f, indent=2)
        logger.info(f"Cached default policies to {cache_path}")
    except Exception as cache_e:
        logger.error(f"Failed to cache default policies to {cache_path}: {cache_e}")

    return default_policies


def initialize_vector_db(orders_df: Optional[pd.DataFrame] = None, use_subset: bool = False):
    """Initializes the vector database, ensuring DataFrame is loaded."""
    # Import here if needed, or ensure it's imported globally if safe
    try:
        from src.vector_db import get_vector_db_instance
    except ImportError as ie:
        logger.error(f"Failed to import get_vector_db_instance: {ie}. Vector DB will not be initialized.")
        return None # Return None or raise error

    if orders_df is None:
        logger.info("No DataFrame provided to initialize_vector_db, loading data...")
        orders_df = load_order_data(use_cache=True) # Load data if not passed

    if orders_df.empty:
         logger.error("Cannot initialize vector DB with an empty DataFrame.")
         return None

    logger.info(f"Initializing vector DB instance with DataFrame (use_subset={use_subset}).")
    try:
        collection = get_vector_db_instance(orders_df, use_subset=use_subset)
        logger.info("Vector DB instance initialized successfully.")
        return collection
    except Exception as db_e:
        logger.error(f"Error initializing vector DB instance: {db_e}", exc_info=True)
        return None # Return None on error
