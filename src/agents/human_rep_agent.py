# src/agents/human_rep_agent.py
import logging
from typing import List, Optional, Dict, Any
from src.agents.base_agent import BaseAgent
from src.core.state import ConversationState
from src.llm.interface import LLMInterface
from src.services.contact_service import save_contact_request # Import the function

logger = logging.getLogger(__name__)

# Define states for the information collection process
STATE_ASK_NAME = "ask_name"
STATE_ASK_EMAIL = "ask_email"
STATE_ASK_PHONE = "ask_phone" # Optional step
STATE_CONFIRM = "confirm"
STATE_COMPLETE = "complete"

# Keys for storing temporary data in conversation state
KEY_HUMAN_REP_STEP = "human_rep_step"
KEY_HUMAN_REP_NAME = "human_rep_name"
KEY_HUMAN_REP_EMAIL = "human_rep_email"
KEY_HUMAN_REP_PHONE = "human_rep_phone"


class HumanRepAgent(BaseAgent):
    """Agent responsible for handling requests to speak to a human representative."""

    agent_name = "human_rep_agent"
    agent_description = "Collects user details (name, email, phone) to create a support ticket for human follow-up."

    def __init__(self, llm_service: LLMInterface, **kwargs):
        super().__init__(llm_service, **kwargs)
        # This agent uses the standalone save_contact_request function.
        # If ContactService was class-based, we'd check for self.contact_service.
        logger.info(f"{self.agent_name} initialized.")

    @staticmethod
    def get_required_service_keys() -> List[str]:
        """Specifies that this agent doesn't require direct service injection."""
        # Uses the standalone save_contact_request function.
        return []

    def _get_current_step(self, state: ConversationState) -> str:
        """Gets the current step in the info collection process from state."""
        return state.extracted_entities.get(KEY_HUMAN_REP_STEP, STATE_ASK_NAME)

    def _update_step(self, state: ConversationState, step: str, data: Optional[Dict[str, Any]] = None):
        """Updates the current step and optionally adds data to state."""
        entities_to_update = {KEY_HUMAN_REP_STEP: step}
        if data:
            entities_to_update.update(data)
        state.update_state(entities=entities_to_update)
        logger.debug(f"Updated human rep state: Step={step}, Data added={list(data.keys()) if data else 'None'}")


    async def process(self, state: ConversationState, user_input: str, **kwargs) -> str:
        """
        Processes user input during the human representative request flow.

        Manages a state machine: Ask Name -> Ask Email -> Ask Phone (Optional) -> Confirm -> Save -> Complete.

        Args:
            state: The current conversation state, used to track progress.
            user_input: The latest message from the user.
            **kwargs: Additional data.

        Returns:
            A string containing the agent's response (asking for info, confirming, or completion message).
        """
        current_step = self._get_current_step(state)
        logger.debug(f"{self.agent_name} processing input. Current step: {current_step}")

        response = "I'm sorry, something went wrong while processing your request for assistance." # Default error

        try:
            # --- State Machine Logic ---
            if current_step == STATE_ASK_NAME:
                # If this is the first step, ask for name.
                # If user_input might contain the name already (less likely if intent was just detected),
                # you could try extracting it here. For simplicity, we assume we need to ask.
                self._update_step(state, STATE_ASK_EMAIL) # Move to next step expectation
                response = "Okay, I can help connect you with a human representative. First, could you please provide your full name?"

            elif current_step == STATE_ASK_EMAIL:
                # Expecting name in user_input from the previous turn.
                # Basic assumption: the user's input IS the name. More robust: use LLM to extract name.
                name = user_input.strip()
                if not name: # Handle empty input
                     response = "Please provide your full name so I can create the request."
                     # Stay in STATE_ASK_EMAIL, but don't save empty name
                else:
                     logger.info(f"Collected name: {name}")
                     self._update_step(state, STATE_ASK_PHONE, data={KEY_HUMAN_REP_NAME: name})
                     response = f"Thanks, {name.split()[0]}! Now, could you please provide your email address?" # Use first name

            elif current_step == STATE_ASK_PHONE:
                # Expecting email in user_input.
                # Basic assumption: user_input IS the email. More robust: use LLM/regex to validate/extract.
                email = user_input.strip()
                # Simple validation (presence of '@')
                if not email or '@' not in email:
                     response = "That doesn't look like a valid email address. Could you please provide your email?"
                     # Stay in STATE_ASK_PHONE, don't save invalid email
                else:
                     logger.info(f"Collected email: {email}")
                     # Ask for optional phone number
                     self._update_step(state, STATE_CONFIRM, data={KEY_HUMAN_REP_EMAIL: email})
                     response = "Got it. Lastly, could you provide a phone number? You can also say 'skip' if you prefer not to."

            elif current_step == STATE_CONFIRM:
                # Expecting phone number or 'skip' in user_input.
                phone = user_input.strip().lower()
                collected_phone = None
                if phone == 'skip':
                    logger.info("User skipped phone number.")
                    phone = None # Ensure it's None if skipped
                else:
                    # Basic check: contains digits. More robust: regex for phone formats.
                    if any(char.isdigit() for char in phone):
                         collected_phone = phone # Store the provided phone number
                         logger.info(f"Collected phone: {collected_phone}")
                    else:
                         logger.warning(f"Potentially invalid phone input: {phone}. Storing as None.")
                         phone = None # Treat potentially invalid input as skipped for now

                # Retrieve collected info from state
                name = state.extracted_entities.get(KEY_HUMAN_REP_NAME)
                email = state.extracted_entities.get(KEY_HUMAN_REP_EMAIL)

                if not name or not email:
                    logger.error("Error: Name or Email missing during confirmation step. Resetting flow.")
                    # Reset the flow
                    self._update_step(state, STATE_ASK_NAME)
                    # Clear potentially stored partial data
                    state.extracted_entities.pop(KEY_HUMAN_REP_NAME, None)
                    state.extracted_entities.pop(KEY_HUMAN_REP_EMAIL, None)
                    state.extracted_entities.pop(KEY_HUMAN_REP_PHONE, None)
                    response = "I seem to have missed some details. Let's start over. Could you please provide your full name?"
                else:
                    # Save the request
                    logger.info(f"Attempting to save contact request: Name={name}, Email={email}, Phone={collected_phone}")
                    success = save_contact_request(
                        full_name=name,
                        email=email,
                        phone_number=collected_phone,
                        notes=f"User requested human assistance via chatbot. Last message: {state.history[-1]['parts'][0] if state.history else 'N/A'}" # Add context
                    )

                    if success:
                        logger.info("Contact request saved successfully.")
                        self._update_step(state, STATE_COMPLETE) # Mark flow as complete
                        response = f"Thank you, {name.split()[0]}! I've created a request with your details (Email: {email}{', Phone: ' + collected_phone if collected_phone else ''}). A member of our team will reach out to you shortly."
                    else:
                        logger.error("Failed to save contact request to the database.")
                        # Don't change state, allow retry or inform user of failure
                        response = "I'm sorry, there was an issue saving your request. Please try asking again in a moment."
                        # Optionally, you could reset the state here if retrying isn't desired.
                        # self._update_step(state, STATE_ASK_NAME) # Reset if needed

            elif current_step == STATE_COMPLETE:
                 # Should ideally not be called again if intent routing is correct,
                 # but handle defensively.
                 logger.debug("Human rep flow already completed for this state.")
                 response = "I've already logged your request for assistance. Our team will be in touch soon!"

        except Exception as e:
            logger.error(f"Error during {self.agent_name} processing (Step: {current_step}): {e}", exc_info=True)
            # Attempt to reset state on unexpected error
            state.extracted_entities.pop(KEY_HUMAN_REP_STEP, None)
            state.extracted_entities.pop(KEY_HUMAN_REP_NAME, None)
            state.extracted_entities.pop(KEY_HUMAN_REP_EMAIL, None)
            state.extracted_entities.pop(KEY_HUMAN_REP_PHONE, None)
            response = "I encountered an unexpected error while handling your request. Please try asking for help again."

        return response