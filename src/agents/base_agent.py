# src/agents/base_agent.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type # Use Type for class references
from src.core.state import ConversationState
from src.llm.interface import LLMInterface
# --- Import service classes themselves, not instances ---
# Uncomment required service classes for type hinting
from src.services.order_service import OrderService
from src.services.policy_service import PolicyService
# Uncomment if ContactService becomes a class
# from src.services.contact_service import ContactService
import logging

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Abstract Base Class for all agents."""

    agent_name: str = "base_agent" # Override in subclasses
    agent_description: str = "Base agent functionality." # Override in subclasses

    def __init__(self, llm_service: LLMInterface, **kwargs):
        """
        Initializes the base agent. Services required by the agent
        should be passed in via kwargs by the ConversationManager.

        Args:
            llm_service: An instance of a class implementing LLMInterface.
            **kwargs: Services needed by the agent (e.g., order_service, policy_service).
                      These are injected by ConversationManager based on get_required_service_keys.
        """
        self.llm_service = llm_service
        # Store references to the services passed in via kwargs
        # Use type hints based on uncommented imports above
        self.order_service: Optional[OrderService] = kwargs.get('order_service')
        self.policy_service: Optional[PolicyService] = kwargs.get('policy_service')
        # Uncomment if ContactService becomes a class
        # self.contact_service: Optional[ContactService] = kwargs.get('contact_service')
        logger.debug(f"Initialized {self.agent_name} with LLM service. Received service kwargs: {list(kwargs.keys())}")


    @staticmethod
    @abstractmethod
    def get_required_service_keys() -> List[str]:
        """
        Returns a list of string keys representing the service instances
        required by this agent (e.g., ['order_service', 'policy_service']).
        The ConversationManager will use these keys to inject the correct
        service instances during agent initialization.
        """
        pass


    @abstractmethod
    async def process(self, state: ConversationState, user_input: str, **kwargs) -> str:
        """
        Processes the user input based on the agent's specific task.

        Args:
            state: The current conversation state.
            user_input: The latest message from the user.
            **kwargs: May include specific data needed for processing,
                      like extracted entities from the ConversationManager.

        Returns:
            A string containing the agent's response.
        """
        pass

    def __str__(self) -> str:
        return f"{self.agent_name}: {self.agent_description}"