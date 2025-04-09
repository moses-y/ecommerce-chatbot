# src/agents/return_policy_agent.py
import logging
from typing import List
from src.agents.base_agent import BaseAgent
from src.core.state import ConversationState
from src.llm.interface import LLMInterface
from src.services.policy_service import PolicyService # Import the class

logger = logging.getLogger(__name__)

class ReturnPolicyAgent(BaseAgent):
    """Agent responsible for providing information about return policies."""

    agent_name = "return_policy_agent"
    agent_description = "Explains the company's return policies based on stored information."

    def __init__(self, llm_service: LLMInterface, **kwargs):
        super().__init__(llm_service, **kwargs)
        self.policy_service: PolicyService = kwargs.get('policy_service')
        if not self.policy_service:
            logger.error(f"{self.agent_name} requires PolicyService but it was not provided.")
            raise ValueError(f"{self.agent_name} requires PolicyService.")
        logger.info(f"{self.agent_name} initialized with PolicyService.")

    @staticmethod
    def get_required_service_keys() -> List[str]:
        """Specifies that this agent requires the PolicyService."""
        return ['policy_service']

    async def process(self, state: ConversationState, user_input: str, **kwargs) -> str:
        """
        Processes user input related to return policies. Retrieves the general policy.
        """
        logger.debug(f"{self.agent_name} processing input: '{user_input}'")

        try:
            # Get the general return policy string using the service
            # Assuming "general_return_policy" is the key for the main policy
            policy_info = self.policy_service.get_policy("general_return_policy")

            if not policy_info: # Handle case where the specific policy isn't found
                 logger.warning("General return policy not found by the service.")
                 policy_info = "I couldn't find the specific return policy information right now."

            logger.info(f"Providing general return policy information.")
            # Ensure policy_info is a string before returning
            if not isinstance(policy_info, str):
                 logger.error(f"Policy service returned non-string type: {type(policy_info)}. Returning error message.")
                 return "Sorry, there was an issue retrieving the policy format."

            return policy_info

        except Exception as e:
            logger.error(f"Error retrieving policy information in {self.agent_name}: {e}", exc_info=True)
            return "Sorry, I encountered an issue retrieving the return policy information at the moment."