# src/agents/order_status_agent.py
import logging
from typing import List, Optional
from src.agents.base_agent import BaseAgent
from src.core.state import ConversationState
from src.llm.interface import LLMInterface
from src.services.order_service import OrderService, format_order_details
from src.utils.helpers import extract_order_id # Make sure this import is correct

logger = logging.getLogger(__name__)

class OrderStatusAgent(BaseAgent):
    """Agent responsible for handling order status inquiries."""

    agent_name = "order_status_agent"
    agent_description = "Checks and reports the status of customer orders using an order ID."

    def __init__(self, llm_service: LLMInterface, **kwargs):
        super().__init__(llm_service, **kwargs)
        # Ensure order_service is injected correctly
        self.order_service = kwargs.get('order_service')
        if not self.order_service:
            logger.error(f"{self.agent_name} requires 'order_service' but it was not provided.")
            # Consider raising an error or handling appropriately
            # raise ValueError(f"{self.agent_name} requires 'order_service'.")
        else:
            # Ensure it's the correct type if needed (optional)
            if not isinstance(self.order_service, OrderService):
                 logger.warning(f"{self.agent_name} received 'order_service' but it might not be the expected type.")
            logger.info(f"{self.agent_name} initialized and received OrderService.")


    @staticmethod
    def get_required_service_keys() -> List[str]:
        """Specifies that this agent requires the 'order_service'."""
        return ['order_service']

    async def process(self, state: ConversationState, user_input: str, **kwargs) -> str:
        """
        Processes user input to check order status.
        Extracts ID, calls service, formats result, or asks for ID.
        """
        logger.debug(f"{self.agent_name} processing input: '{user_input}'")
        order_id: Optional[str] = None

        # 1. Check state (optional)
        # ...

        # 2. Try extracting from current input
        if not order_id:
            # --- ADD DETAILED LOGGING HERE ---
            logger.info(f"Attempting to extract order ID from user_input: type={type(user_input)}, value='{user_input}'")
            # --- END ADDED LOGGING ---
            order_id = extract_order_id(user_input) # Call the helper function
            if order_id:
                logger.info(f"Extracted order ID from user input: {order_id}")
                # Optional: Store extracted ID in state
                # state.update_state(entities={'order_id': order_id})
            else:
                logger.info("No order ID found in state or current input.") # This log might be misleading if extraction failed unexpectedly

        # 3. If Order ID is available, get status
        if order_id:
            logger.info(f"Querying order status for ID: {order_id} using OrderService instance.")
            if not self.order_service:
                logger.error(f"{self.agent_name} cannot process: OrderService not available.")
                return "Sorry, the order checking service is currently unavailable."

            try:
                order_data = await self.order_service.get_order_status_by_id(order_id)

                if order_data:
                    logger.info(f"Order found for ID {order_id}. Formatting details.")
                    formatted_response = format_order_details(order_data)
                    return formatted_response
                else:
                    logger.warning(f"Order ID {order_id} not found by OrderService.")
                    return f"Sorry, I couldn't find any order with the ID '{order_id}'. Please double-check the ID."

            except Exception as e:
                logger.error(f"Error calling OrderService for ID {order_id}: {e}", exc_info=True)
                return "Sorry, I encountered an error while checking the order status."

        # 4. If no Order ID found, ask for it
        else:
            # This path is taken if extract_order_id returned None
            logger.info("Order ID extraction returned None. Asking user for it.")
            return "Okay, I can help with that. Please provide the 32-character alphanumeric order ID found in your confirmation email."