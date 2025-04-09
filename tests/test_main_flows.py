# tests/test_main_flows.py (Rolled back + 32-char Fix Applied)
import pytest
from unittest.mock import MagicMock, AsyncMock, call

from src.agents.order_status_agent import OrderStatusAgent # Add this import if not already there
from src.core.state import ConversationState # Add this import

# Import necessary components and helpers
from src.core.conversation import ConversationManager # is imported if needed
from src.services.order_service import format_order_details
# We don't need extract_order_id here anymore for the agent tests
from src.utils.helpers import extract_order_id
from src.core.config import ORDER_STATUS_DESCRIPTIONS

# Apply asyncio mark only to async tests via decorator

# --- Conversation Manager and Intent Routing Tests ---
# These should pass without issue as they don't depend heavily on the failing parts

@pytest.mark.asyncio
async def test_conversation_manager_greeting(conversation_manager, test_session_id):
    """Test the initial state or greeting (if applicable)."""
    assert conversation_manager is not None

@pytest.mark.asyncio
async def test_intent_routing_order_status(conversation_manager, mock_llm_service, test_session_id):
    """Test routing to OrderStatusAgent when intent is check_order_status."""
    user_input = "check my order"
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert "Please provide the 32-character alphanumeric order ID" in response_data["response"]
    # Check the call signature correctly
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input
    # Add checks for other args if necessary, e.g.,
    # assert 'available_intents' in kwargs
    # assert 'history' in kwargs

@pytest.mark.asyncio
async def test_intent_routing_return_policy(conversation_manager, mock_llm_service, mock_policy_service, test_session_id):
    """Test routing to ReturnPolicyAgent."""
    user_input = "what's your return policy?"
    mock_llm_service.determine_intent.return_value = 'ask_return_policy'

    expected_policy = "Test return policy text."
    mock_policy_service.get_policy.return_value = expected_policy

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert response_data["response"] == expected_policy
    mock_policy_service.get_policy.assert_called_once()
    # Check the call signature correctly
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input


@pytest.mark.asyncio
async def test_intent_routing_request_human(conversation_manager, mock_llm_service, test_session_id):
    """Test routing to HumanRepAgent."""
    user_input = "talk to a person"
    mock_llm_service.determine_intent.return_value = 'request_human'

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert "Okay, I can help connect you" in response_data["response"]
    # Check the call signature correctly
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input


@pytest.mark.asyncio
async def test_intent_routing_unknown(conversation_manager, mock_llm_service, test_session_id):
    """Test fallback response for unknown intent using LLM generation."""
    user_input = "tell me a joke"
    mock_llm_service.determine_intent.return_value = 'unknown'
    expected_fallback = "Mock LLM: Sorry, I can't tell jokes right now."
    mock_llm_service.generate_response.return_value = expected_fallback

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert response_data["response"] == expected_fallback
    # Check the call signature correctly for determine_intent
    mock_llm_service.determine_intent.assert_called_once()
    args_intent, kwargs_intent = mock_llm_service.determine_intent.call_args
    assert kwargs_intent.get('user_input') == user_input
    # Check the call signature correctly for generate_response (assuming similar pattern)
    mock_llm_service.generate_response.assert_called_once()
    args_gen, kwargs_gen = mock_llm_service.generate_response.call_args
    assert kwargs_gen.get('prompt') is not None # Or check specific prompt content if needed
    assert 'history' in kwargs_gen


# --- Agent Interaction Tests (Focus of Fixes) ---

@pytest.mark.asyncio
async def test_order_status_agent_found(conversation_manager: ConversationManager, mock_llm_service: AsyncMock, mock_order_service: AsyncMock, sample_order_data_found: MagicMock, test_session_id: str):
    """Test OrderStatusAgent response when order is found (simulating 2 steps)."""
    # --- FIXED: Use 32-char ID from updated fixture ---
    order_id = sample_order_data_found.order_id
    assert len(order_id) == 32 # Verify length

    # --- Step 1: User asks to check status (Agent should ask for ID) ---
    user_input_1 = "check my order status"
    mock_llm_service.determine_intent.reset_mock()
    mock_order_service.get_order_status_by_id.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data_1 = await conversation_manager.handle_message(user_input_1, test_session_id)

    assert "Please provide the 32-character alphanumeric order ID" in response_data_1["response"]
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input_1
    assert isinstance(kwargs.get('available_intents'), list)
    assert isinstance(kwargs.get('history'), list)
    mock_order_service.get_order_status_by_id.assert_not_called()

    # --- Step 2: User provides ONLY the order ID ---
    user_input_2 = order_id # Pass the 32-char ID
    mock_llm_service.determine_intent.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status' # Assume intent re-check needed
    # Configure mock service to return the *fixture object* when called with the correct ID
    mock_order_service.get_order_status_by_id.return_value = sample_order_data_found

    response_data_2 = await conversation_manager.handle_message(user_input_2, test_session_id)

    # Expected response is formatting the fixture object
    expected_formatted_details = format_order_details(sample_order_data_found)
    # --- Should PASS now: Agent extracts 32-char ID, calls service, gets mock, formats ---
    assert response_data_2["response"] == expected_formatted_details
    mock_order_service.get_order_status_by_id.assert_called_once_with(order_id)
    mock_llm_service.determine_intent.assert_called_once() # Check intent was determined again
    args_step2, kwargs_step2 = mock_llm_service.determine_intent.call_args
    assert kwargs_step2.get('user_input') == user_input_2
    assert isinstance(kwargs_step2.get('available_intents'), list)
    assert isinstance(kwargs_step2.get('history'), list)
    assert len(kwargs_step2.get('history', [])) > 0
    assert kwargs_step2['history'][0]['parts'][0] == user_input_1


@pytest.mark.asyncio
async def test_order_status_agent_not_found(conversation_manager: ConversationManager, mock_llm_service: AsyncMock, mock_order_service: AsyncMock, test_session_id: str):
    """Test OrderStatusAgent response when order is not found (simulating 2 steps)."""
    # --- FIXED: Use 32-char non-existent ID ---
    order_id = "11111111111111111111111111111111" # Example 32-char non-existent ID
    assert len(order_id) == 32

    # --- Step 1: User asks to check status (Agent should ask for ID) ---
    user_input_1 = "where is my order"
    mock_llm_service.determine_intent.reset_mock()
    mock_order_service.get_order_status_by_id.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data_1 = await conversation_manager.handle_message(user_input_1, test_session_id)

    assert "Please provide the 32-character alphanumeric order ID" in response_data_1["response"]
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input_1
    assert isinstance(kwargs.get('available_intents'), list)
    assert isinstance(kwargs.get('history'), list)
    mock_order_service.get_order_status_by_id.assert_not_called()

    # --- Step 2: User provides ONLY the (non-existent) order ID ---
    user_input_2 = order_id # Pass the 32-char non-existent ID
    mock_llm_service.determine_intent.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status' # Assume intent re-check
    # Configure mock service to return None for this ID
    mock_order_service.get_order_status_by_id.return_value = None

    response_data_2 = await conversation_manager.handle_message(user_input_2, test_session_id)

    # --- Should PASS now: Agent extracts 32-char ID, calls service, gets None, returns not found msg ---
    expected_not_found_msg = f"Sorry, I couldn't find any order with the ID '{order_id}'. Please double-check the ID."
    assert expected_not_found_msg in response_data_2["response"] # Use 'in' for flexibility
    mock_order_service.get_order_status_by_id.assert_called_once_with(order_id)
    mock_llm_service.determine_intent.assert_called_once() # Check intent was determined again
    args_step2, kwargs_step2 = mock_llm_service.determine_intent.call_args
    assert kwargs_step2.get('user_input') == user_input_2
    assert isinstance(kwargs_step2.get('available_intents'), list)
    assert isinstance(kwargs_step2.get('history'), list)
    assert len(kwargs_step2.get('history', [])) > 0
    assert kwargs_step2['history'][0]['parts'][0] == user_input_1


# --- Helper Function Tests (Synchronous) ---

def test_format_order_details_delivered(sample_order_data_found):
    """Test formatting for a delivered order."""
    # --- FIXED: Assert using the 32-char ID from the updated fixture ---
    expected_id = "ayc123def456ghi789jkl012mno345p7" # 32 chars
    assert sample_order_data_found.order_id == expected_id # Verify fixture ID itself
    formatted_details = format_order_details(sample_order_data_found)
    assert f"Order ID: {expected_id}" in formatted_details
    assert ORDER_STATUS_DESCRIPTIONS['delivered'] in formatted_details
    assert "Purchased on: 2025-04-01 10:30" in formatted_details
    assert "Estimated Delivery: 2025-04-08" in formatted_details
    assert "Delivered on: 2025-04-07 14:00" in formatted_details

def test_format_order_details_invoiced(sample_order_data_invoiced):
    """Test formatting for an invoiced (not delivered) order."""
    # --- FIXED: Assert using the 32-char ID from the updated fixture ---
    expected_id = "xyz987abc654def321ghi098jkl7650a" # 32 chars
    assert sample_order_data_invoiced.order_id == expected_id # Verify fixture ID itself
    formatted_details = format_order_details(sample_order_data_invoiced)
    assert f"Order ID: {expected_id}" in formatted_details
    assert ORDER_STATUS_DESCRIPTIONS['invoiced'] in formatted_details
    assert "Purchased on: 2025-04-08 11:00" in formatted_details
    assert "Estimated Delivery: 2025-04-15" in formatted_details
    assert "Delivered on:" not in formatted_details # Correct for invoiced

def test_format_order_details_none():
    """Test formatting when None is passed."""
    formatted = format_order_details(None)
    assert formatted == "Order details could not be retrieved."


# --- NEW Direct Agent Test ---
@pytest.mark.asyncio
async def test_order_status_agent_process_direct_id(
    mock_llm_service: AsyncMock,
    mock_order_service: AsyncMock,
    sample_order_data_found: MagicMock,
    test_session_id: str # Use the session ID fixture
):
    """Test OrderStatusAgent.process directly when input is just the ID."""
    # Instantiate the agent directly, injecting the mock service
    # Make sure the agent's __init__ correctly receives and stores the service
    agent = OrderStatusAgent(llm_service=mock_llm_service, order_service=mock_order_service)

    # Create a dummy state object for the agent call
    state = ConversationState(session_id=test_session_id)

    # The input is just the ID from the fixture
    order_id = sample_order_data_found.order_id
    user_input = order_id
    assert len(user_input) == 32 # Verify length

    # Configure the mock service to return the data when called with this ID
    mock_order_service.get_order_status_by_id.reset_mock() # Reset before call
    mock_order_service.get_order_status_by_id.return_value = sample_order_data_found

    # Call the agent's process method directly
    response = await agent.process(state, user_input)

    # Assert the service was called correctly
    mock_order_service.get_order_status_by_id.assert_called_once_with(order_id)

    # Assert the response is the formatted order details
    expected_formatted_details = format_order_details(sample_order_data_found)
    assert response == expected_formatted_details
# --- End New Test ---
# Keep test_extract_order_id commented out for now
# ...


# The test_extract_order_id function and its parametrize decorator has been commented out
# Unique issue with regex.

@pytest.mark.parametrize("text, expected_id", [
    # Corrected: Added '0' to make IDs 32 chars long
    ("my order id is abc123def456ghi789jkl012mno345p0 please check", "abc123def456ghi789jkl012mno345p0"),
    ("abc123def456ghi789jkl012mno345p0", "abc123def456ghi789jkl012mno345p0"),
    ("check status for xyz987abc654def321ghi098jkl7650a thanks", "xyz987abc654def321ghi098jkl7650a"),
    ("order 12345", None), # Too short - Remains the same
    ("order abc123def456ghi789jkl012mno345p0!", "abc123def456ghi789jkl012mno345p0"), # Corrected ID
    ("no order id here", None), # Remains the same
    ("", None), # Remains the same
    # These used valid 32-char IDs already
    ("The ID is e481f51cbdc54678b7cc49136f2d6af7.", "e481f51cbdc54678b7cc49136f2d6af7"),
    ("53cdb2fc8bc7dce0b6741e2150273451 is the order", "53cdb2fc8bc7dce0b6741e2150273451"),
])
def test_extract_order_id(text, expected_id):
    """Test the order ID extraction helper."""
    # Add print statements here if needed for debugging the test itself
    # print(f"Testing text: '{text}' | Expecting: '{expected_id}'")
    # extracted = extract_order_id(text)
    # print(f"Extracted: '{extracted}'")
    assert extract_order_id(text) == expected_id

