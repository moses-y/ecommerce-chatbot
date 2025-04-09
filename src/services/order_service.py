# src/services/order_service.py
import logging
from sqlalchemy.orm import Session
from src.db.database import SessionLocal, get_db # Assuming get_db yields a session
from src.db.models import Order
from src.core.config import ORDER_STATUS_DESCRIPTIONS
import asyncio # Needed for running sync code in async method if necessary

logger = logging.getLogger(__name__)

# --- Helper Function ---
def format_order_details(order: Order | None) -> str:
    """Formats order details into a user-friendly string."""
    if not order:
        # Return a generic message if None is passed explicitly
        return "Order details could not be retrieved."

    status_desc = ORDER_STATUS_DESCRIPTIONS.get(order.order_status, f"Status: {order.order_status}")
    details = [f"Order ID: {order.order_id}", status_desc]

    # Use getattr with default None to safely access optional date fields
    purchase_ts = getattr(order, 'order_purchase_timestamp', None)
    estimated_delivery_ts = getattr(order, 'order_estimated_delivery_date', None)
    delivered_ts = getattr(order, 'order_delivered_customer_date', None)

    if purchase_ts:
        try:
            details.append(f"Purchased on: {purchase_ts.strftime('%Y-%m-%d %H:%M')}")
        except AttributeError:
             details.append(f"Purchased on: {purchase_ts}") # Fallback if not datetime
    if estimated_delivery_ts:
         try:
            details.append(f"Estimated Delivery: {estimated_delivery_ts.strftime('%Y-%m-%d')}")
         except AttributeError:
             details.append(f"Estimated Delivery: {estimated_delivery_ts}") # Fallback

    if delivered_ts:
        try:
            details.append(f"Delivered on: {delivered_ts.strftime('%Y-%m-%d %H:%M')}")
        except AttributeError:
            details.append(f"Delivered on: {delivered_ts}") # Fallback

    return "\n".join(details)


# --- Service Class ---
class OrderService:
    """Service class for handling order-related operations."""

    # If your service needs persistent resources (like a DB pool),
    # you might inject them via __init__. For now, we use SessionLocal per call.

    async def get_order_status_by_id(self, order_id: str) -> str:
        """
        Fetches order status and details by order ID from the database asynchronously.

        Args:
            order_id: The 32-character order ID.

        Returns:
            A formatted string with order details or a 'not found'/'error' message.
        """
        logger.info(f"Attempting to fetch order details for order_id: {order_id}")

        # Basic validation
        if not order_id or len(order_id) != 32 or not order_id.isalnum():
            logger.warning(f"Invalid order ID format received: {order_id}")
            # Return a specific message for invalid format
            return "The provided order ID seems invalid. Please provide a 32-character alphanumeric ID."

        db: Session = SessionLocal()
        try:
            # Run the synchronous database query in a separate thread
            # to avoid blocking the asyncio event loop.
            order = await asyncio.to_thread(
                db.query(Order).filter(Order.order_id == order_id).first
            )

            if order:
                logger.info(f"Order found for ID {order_id}. Status: {order.order_status}")
                # Use the standalone helper function for formatting
                return format_order_details(order)
            else:
                logger.warning(f"No order found for ID: {order_id}")
                return f"Sorry, I couldn't find any order with the ID '{order_id}'. Please double-check the ID."

        except Exception as e:
            logger.error(f"Database error fetching order {order_id}: {e}", exc_info=True)
            return "Sorry, I encountered an error while trying to retrieve the order details. Please try again later."
        finally:
            # Ensure the session is closed even if errors occur
            await asyncio.to_thread(db.close)
            logger.debug(f"Database session closed for order query: {order_id}")