# utils.py
import os
import sys
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import json
from typing import Tuple, Dict, Any, Optional
from datetime import datetime
from config import ORDER_STATUS_DESCRIPTIONS

def load_order_data(use_cache: bool = True) -> pd.DataFrame:
    """
    Load order data from CSV file or cached version.

    Args:
        use_cache: Whether to use cached data if available

    Returns:
        DataFrame containing order data
    """
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    cache_path = os.path.join("data", "cached_orders.csv")
    original_path = os.path.join("data", "olist_orders_dataset.csv")

    # Try to load from cache first if enabled
    if use_cache and os.path.exists(cache_path):
        try:
            return pd.read_csv(cache_path)
        except Exception as e:
            print(f"Error loading cached data: {e}")

    # Load original data
    if os.path.exists(original_path):
        orders_df = pd.read_csv(original_path)

        # Create cache for future use
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        orders_df.to_csv(cache_path, index=False)

        return orders_df
    else:
        raise FileNotFoundError(f"Order dataset not found at {original_path}")

def create_order_index(orders_df):
    """Create indexed dictionaries for faster order lookups."""
    order_index = {}
    customer_index = {}

    for _, row in orders_df.iterrows():
        order_id = row['order_id']
        customer_id = row.get('customer_id')
        order_data = row.to_dict()

        # Index by order ID
        order_index[order_id] = order_data

        # Index by customer ID
        if customer_id:
            if customer_id not in customer_index:
                customer_index[customer_id] = []
            customer_index[customer_id].append(order_data)

    return order_index, customer_index

def get_order_status(orders_df: pd.DataFrame, order_id: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Get the status and details of an order.

    Args:
        orders_df: DataFrame containing order data
        order_id: ID of the order to look up

    Returns:
        Tuple of (status, details) where details is a dictionary of order information
    """
    # Filter for the specific order
    order = orders_df[orders_df['order_id'] == order_id]

    if order.empty:
        return None, None

    # Extract order details
    order_row = order.iloc[0]
    status = order_row['order_status']

    # Format dates for better readability
    purchase_date = pd.to_datetime(order_row['order_purchase_timestamp']).strftime('%B %d, %Y') if pd.notna(order_row['order_purchase_timestamp']) else "Not available"
    approved_date = pd.to_datetime(order_row['order_approved_at']).strftime('%B %d, %Y') if pd.notna(order_row['order_approved_at']) else "Not available"

    # Calculate estimated delivery if available
    estimated_delivery = None
    if pd.notna(order_row['order_estimated_delivery_date']):
        est_delivery_date = pd.to_datetime(order_row['order_estimated_delivery_date'])
        estimated_delivery = est_delivery_date.strftime('%B %d, %Y')

        # Check if delivery is overdue
        if est_delivery_date < datetime.now() and status != 'delivered':
            estimated_delivery += " (Overdue)"

    # Actual delivery date if available
    actual_delivery = pd.to_datetime(order_row['order_delivered_customer_date']).strftime('%B %d, %Y') if pd.notna(order_row['order_delivered_customer_date']) else "Not yet delivered"

    details = {
        'purchase_date': purchase_date,
        'approved_date': approved_date,
        'estimated_delivery': estimated_delivery,
        'actual_delivery': actual_delivery
    }

    return status, details

def format_order_details(order_id: str, status: str, details: Dict[str, Any]) -> str:
    """
    Format order details into a human-readable response.

    Args:
        order_id: ID of the order
        status: Current status of the order
        details: Dictionary containing order details

    Returns:
        Formatted string with order information
    """
    # ORDER_STATUS_DESCRIPTIONS is imported at the top of the file
    status_description = ORDER_STATUS_DESCRIPTIONS.get(status, "Unknown status")

    response = f"Order #{order_id} is currently {status}.\n\n"
    response += f"{status_description}\n\n"
    response += f"Purchase date: {details['purchase_date']}\n"

    if status != 'canceled':
        if details.get('approved_date') and details['approved_date'] != "Not available":
            response += f"Approved date: {details['approved_date']}\n"

        if details.get('estimated_delivery'):
            response += f"Estimated delivery: {details['estimated_delivery']}\n"

        if status == 'delivered' and details.get('actual_delivery'):
            response += f"Delivered on: {details['actual_delivery']}\n"

    # Add additional helpful information based on status
    if status == 'processing':
        response += "\nYour order is being processed. You'll receive a confirmation email once it ships."
    elif status == 'shipped':
        response += "\nYour order is on its way! You can track it using the tracking number in your shipping confirmation email."
    elif status == 'canceled':
        response += "\nIf you have questions about this cancellation, please contact our customer service."
    elif status == 'delivered' and details.get('estimated_delivery') and "Overdue" in details['estimated_delivery']:
        response += "\nWe apologize for the delay in delivery."

    return response

def load_return_policies(use_cache: bool = True) -> Dict[str, Any]:
    """
    Load return policies from JSON file or cached version.

    Args:
        use_cache: Whether to use cached data if available

    Returns:
        Dictionary containing return policies
    """
    cache_path = os.path.join("data", "cached_policies.json")

    # Try to load from cache if enabled
    if use_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading cached policies: {e}")

    # Define default policies
    policies = {
        "standard": {
            "days": 30,
            "condition": "unused and in original packaging",
            "process": "Complete the return form on our website and use the prepaid shipping label."
        },
        "electronics": {
            "days": 14,
            "condition": "unopened or defective",
            "process": "Contact customer support for a return authorization before shipping."
        },
        "clothing": {
            "days": 45,
            "condition": "unworn with tags attached",
            "process": "Return to any of our physical stores or ship back using our return portal."
        },
        "perishable": {
            "days": 3,
            "condition": "damaged or spoiled on arrival",
            "process": "Take photos of the items and contact customer support immediately."
        }
    }

    # Create cache for future use
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'w') as f:
        json.dump(policies, f, indent=2)

    return policies