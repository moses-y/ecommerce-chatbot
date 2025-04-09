# src/core/conversation.py
import logging
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any # Added Any
from src.core.state import ConversationState
from src.llm.interface import LLMInterface
# from src.llm.gemini_service import GeminiService # Keep import generic if possible
from src.agents.base_agent import BaseAgent
from src.agents.order_status_agent import OrderStatusAgent
from src.agents.return_policy_agent import ReturnPolicyAgent
from src.agents.human_rep_agent import HumanRepAgent, KEY_HUMAN_REP_STEP, STATE_COMPLETE, STATE_ASK_NAME # Import state keys

# --- Import Service Classes ---
from src.services.policy_service import PolicyService
# Import the OrderService CLASS
from src.services.order_service import OrderService
# Import ContactService if it becomes a class, otherwise remove comment
# from src.services.contact_service import ContactService

from src.core.config import MAX_CONVERSATION_HISTORY

logger = logging.getLogger(__name__)

class ConversationManager:
    """
    Manages conversation state, intent detection, and agent routing.
    """
    def __init__(self, llm_service: LLMInterface):
        """
        Initializes the ConversationManager with the LLM service and agents.

        Args:
            llm_service: An instance of a class implementing LLMInterface (e.g., GeminiService).
        """
        self.llm_service = llm_service
        self.conversation_states: Dict[str, ConversationState] = {} # Stores state per session_id

        # --- Instantiate required services ---
        # Singleton pattern in PolicyService means this gets the single instance
        self.policy_service = PolicyService()
        # Instantiate OrderService CLASS
        self.order_service = OrderService()
        # Instantiate ContactService if it becomes a class
        # self.contact_service = ContactService()
        logger.info("Core services instantiated.")

        # --- Prepare available services for injection ---
        self.available_services: Dict[str, Any] = {
            'policy_service': self.policy_service,
            # Add the order_service instance
            'order_service': self.order_service,
            # Add contact_service if it's a class instance
            # 'contact_service': self.contact_service,
        }
        logger.info(f"Available services for agent injection: {list(self.available_services.keys())}")

        # Instantiate agents and map them to intents
        self.agents: Dict[str, BaseAgent] = self._load_agents()
        self.intents: List[str] = list(self.agents.keys())
        logger.info(f"ConversationManager initialized with intents: {self.intents}")


    def _load_agents(self) -> Dict[str, BaseAgent]:
        """Instantiates and returns a dictionary of agents mapped to their intents."""
        agents = {}
        agent_classes = [OrderStatusAgent, ReturnPolicyAgent, HumanRepAgent]
        intent_map = {
            # Map class to intent string
            OrderStatusAgent: "check_order_status",
            ReturnPolicyAgent: "ask_return_policy",
            HumanRepAgent: "request_human",
        }

        for agent_cls in agent_classes:
            intent = intent_map.get(agent_cls)
            if not intent:
                 logger.warning(f"No intent mapping found for agent class {agent_cls.__name__}. Skipping.")
                 continue

            try:
                required_keys = agent_cls.get_required_service_keys()
                logger.debug(f"Agent {agent_cls.__name__} requires services: {required_keys}")
                services_for_agent = {
                    key: self.available_services[key]
                    for key in required_keys if key in self.available_services
                }

                # Check if all required services were found
                if len(services_for_agent) != len(required_keys):
                     missing = set(required_keys) - set(services_for_agent.keys())
                     logger.error(f"Cannot initialize agent {agent_cls.__name__}. Missing required services: {missing}. Available: {list(self.available_services.keys())}")
                     continue # Skip this agent

                # Instantiate agent, passing services as keyword arguments
                agents[intent] = agent_cls(llm_service=self.llm_service, **services_for_agent)
                logger.info(f"Successfully loaded agent '{agent_cls.agent_name}' for intent '{intent}' with services: {list(services_for_agent.keys())}")

            except Exception as e:
                logger.error(f"Failed to initialize agent {agent_cls.__name__}: {e}", exc_info=True)

        return agents


    def _get_or_create_state(self, session_id: Optional[str] = None) -> ConversationState:
        """Retrieves or creates a ConversationState for a given session ID."""
        if session_id is None:
            session_id = str(uuid.uuid4())
            logger.info(f"No session ID provided. Created new session: {session_id}")

        if session_id not in self.conversation_states:
            logger.info(f"Creating new conversation state for session: {session_id}")
            self.conversation_states[session_id] = ConversationState(session_id=session_id)
        else:
             logger.debug(f"Found existing conversation state for session: {session_id}")

        # Trim history if it exceeds the maximum length
        state = self.conversation_states[session_id]
        if len(state.history) > MAX_CONVERSATION_HISTORY:
             logger.debug(f"Trimming history for session {session_id} from {len(state.history)} to {MAX_CONVERSATION_HISTORY}")
             state.history = state.history[-MAX_CONVERSATION_HISTORY:]

        return state

    async def handle_message(self, user_input: str, session_id: Optional[str] = None) -> Dict[str, str]:
        """
        Handles an incoming user message, determines intent, routes to an agent,
        and returns the response.

        Args:
            user_input: The text message from the user.
            session_id: The unique identifier for the conversation session. If None, a new one is created.

        Returns:
            A dictionary containing the 'response' string and the 'session_id'.
        """
        state = self._get_or_create_state(session_id)
        session_id = state.session_id # Ensure we have the definitive session ID

        logger.info(f"Handling message for session {session_id}: '{user_input[:50]}...'")

        # Add user message to state
        state.add_message(role="user", text=user_input)

        # --- Determine Agent ---
        selected_agent: Optional[BaseAgent] = None
        intent: Optional[str] = None

        # Check if HumanRepAgent flow is active and not completed/just started
        human_rep_step = state.extracted_entities.get(KEY_HUMAN_REP_STEP)
        # Check if the intent is already set to request_human (e.g., from previous turn asking for name)
        is_human_rep_intent = state.current_intent == "request_human"

        # Prioritize active multi-turn flows like human rep request
        if human_rep_step and human_rep_step not in [STATE_ASK_NAME, STATE_COMPLETE]:
            logger.info(f"Human representative flow active (Step: {human_rep_step}). Routing directly to HumanRepAgent.")
            intent = "request_human" # Force intent
            selected_agent = self.agents.get(intent)
            state.update_state(intent=intent) # Ensure intent is set for this turn
        else:
            # If no specific flow active, determine intent
            logger.debug("Determining intent using LLM service...")
            # Use determine_intent which might be simpler than determine_intent_and_entities
            detected_intent = await self.llm_service.determine_intent( # Assuming determine_intent is async
                user_input=user_input,
                available_intents=self.intents,
                history=state.get_history() # Provide history for context
            )
            logger.info(f"Detected intent: '{detected_intent}'")
            state.update_state(intent=detected_intent) # Store detected intent in state
            intent = detected_intent # Use the detected intent

            # Select agent based on intent
            selected_agent = self.agents.get(intent)

        # --- Execute Agent or Default Response ---
        bot_response: str

        if selected_agent:
            logger.info(f"Routing to agent: {selected_agent.agent_name}")
            try:
                # Pass relevant state and input to the agent
                bot_response = await selected_agent.process(state=state, user_input=user_input)
                state.update_state(agent=selected_agent.agent_name) # Record which agent handled it
                logger.debug(f"Agent '{selected_agent.agent_name}' response: '{bot_response[:100]}...'")
            except Exception as e:
                logger.error(f"Error processing message with agent {selected_agent.agent_name}: {e}", exc_info=True)
                bot_response = "I'm sorry, I encountered an error trying to handle your request. Could you please try rephrasing?"
                # Optionally reset specific state if agent failed mid-flow
                if intent == "request_human":
                     state.extracted_entities.pop(KEY_HUMAN_REP_STEP, None) # Reset human rep flow on error

        elif intent == 'general_query' or intent == 'unknown':
            logger.info("Handling as general query using LLM.")
            # Use the LLM's general generation capability
            bot_response = await self.llm_service.generate_response( # Assuming generate_response is async
                prompt=user_input, # Pass user input directly
                history=state.get_history() # Provide history
            )
            state.update_state(agent='llm_general_response')
        else:
            # This case should ideally not happen if intents cover all agents + general/unknown
            logger.warning(f"No agent found for intent '{intent}' and not general/unknown. Providing default response.")
            bot_response = "I'm sorry, I'm not sure how to handle that specific request. Can I help with order status, return policies, or connecting you to a human representative?"
            state.update_state(agent='fallback_response')


        # Add bot response to state
        state.add_message(role="model", text=bot_response)

        # Clean up transient state elements if needed (e.g., intent for the *next* turn)
        # state.clear_transient_state() # Decide if this is needed

        logger.info(f"Final response for session {session_id}: '{bot_response[:100]}...'")

        # Return response and session ID
        return {"response": bot_response, "session_id": session_id}

    # --- Optional: Cleanup method ---
    def cleanup_inactive_sessions(self, max_age_seconds: int = 3600):
        """Removes conversation states that haven't been active for a while."""
        now = datetime.now() # Use datetime.datetime.now()
        inactive_sessions = [
            sid for sid, state in self.conversation_states.items()
            if (now - state.last_interaction_time).total_seconds() > max_age_seconds
        ]
        if inactive_sessions:
            logger.info(f"Cleaning up {len(inactive_sessions)} inactive sessions older than {max_age_seconds} seconds.")
            for sid in inactive_sessions:
                del self.conversation_states[sid]