# tests/test_main_flows.py
import pytest
from unittest.mock import MagicMock, AsyncMock, call
import datetime # Ensure datetime is imported if used in fixtures/formatting

# Import necessary components from your application
from src.core.conversation import ConversationManager
from src.core.state import ConversationState
from src.agents.order_status_agent import OrderStatusAgent
from src.services.order_service import format_order_details
from src.utils.helpers import extract_order_id # Keep if test_extract_order_id is used
from src.core.config import ORDER_STATUS_DESCRIPTIONS

# --- Conversation Manager and Intent Routing Tests ---

@pytest.mark.asyncio
async def test_conversation_manager_greeting(conversation_manager, test_session_id):
    """Test the initial state or greeting (if applicable)."""
    # Basic check that the manager fixture is available
    assert conversation_manager is not None
    # If there's an initial greeting message, test it here.
    # Example: initial_state = conversation_manager.get_state(test_session_id)
    # assert initial_state.history == [] # Or check for initial greeting message

@pytest.mark.asyncio
async def test_intent_routing_order_status(conversation_manager, mock_llm_service, test_session_id):
    """Test routing to OrderStatusAgent when intent is check_order_status."""
    user_input = "check my order"
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    # Expect the agent to ask for the ID
    assert "Please provide the 32-character alphanumeric order ID" in response_data["response"]
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input
    assert isinstance(kwargs.get('available_intents'), list)
    assert isinstance(kwargs.get('history'), list)

@pytest.mark.asyncio
async def test_intent_routing_return_policy(conversation_manager, mock_llm_service, mock_policy_service, test_session_id):
    """Test routing to ReturnPolicyAgent."""
    user_input = "what's your return policy?"
    mock_llm_service.determine_intent.return_value = 'ask_return_policy'
    expected_policy = "Mock return policy text from service."
    mock_policy_service.get_policy.return_value = expected_policy # Mock the service directly

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert response_data["response"] == expected_policy
    mock_policy_service.get_policy.assert_called_once()
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input

@pytest.mark.asyncio
async def test_intent_routing_request_human(conversation_manager, mock_llm_service, test_session_id):
    """Test routing to HumanRepAgent."""
    user_input = "talk to a person"
    mock_llm_service.determine_intent.return_value = 'request_human'

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    # Expect the agent to start the information gathering process
    assert "Okay, I can help connect you" in response_data["response"]
    assert "please provide your full name" in response_data["response"].lower() # Check for first question
    mock_llm_service.determine_intent.assert_called_once()
    args, kwargs = mock_llm_service.determine_intent.call_args
    assert kwargs.get('user_input') == user_input

@pytest.mark.asyncio
async def test_intent_routing_unknown(conversation_manager, mock_llm_service, test_session_id):
    """Test fallback response for unknown intent using LLM generation."""
    user_input = "tell me a joke"
    mock_llm_service.determine_intent.return_value = 'unknown'
    expected_fallback = "Mock LLM: Sorry, I cannot fulfill that request."
    mock_llm_service.generate_response.return_value = expected_fallback

    response_data = await conversation_manager.handle_message(user_input, test_session_id)

    assert response_data["response"] == expected_fallback
    mock_llm_service.determine_intent.assert_called_once()
    args_intent, kwargs_intent = mock_llm_service.determine_intent.call_args
    assert kwargs_intent.get('user_input') == user_input

    mock_llm_service.generate_response.assert_called_once()
    args_gen, kwargs_gen = mock_llm_service.generate_response.call_args
    assert kwargs_gen.get('prompt') is not None # Check that a prompt was generated
    assert isinstance(kwargs_gen.get('history'), list) # Check history was passed

# --- Agent Interaction Tests ---

@pytest.mark.asyncio
async def test_order_status_agent_found(
    conversation_manager: ConversationManager,
    mock_llm_service: AsyncMock,
    mock_order_service: AsyncMock,
    sample_order_data_found: MagicMock, # Assumes this fixture provides a valid Order object
    test_session_id: str
):
    """Test OrderStatusAgent response when order is found (simulating 2 steps)."""
    order_id = sample_order_data_found.order_id
    assert isinstance(order_id, str) and len(order_id) == 32 # Verify ID format from fixture

    # --- Step 1: User asks to check status -> Agent asks for ID ---
    user_input_1 = "check my order status"
    mock_llm_service.determine_intent.reset_mock()
    mock_order_service.get_order_status_by_id.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data_1 = await conversation_manager.handle_message(user_input_1, test_session_id)

    assert "Please provide the 32-character alphanumeric order ID" in response_data_1["response"]
    mock_llm_service.determine_intent.assert_called_once_with(
        user_input=user_input_1,
        available_intents=conversation_manager.available_intents,
        history=[] # Assuming history is empty initially for this session
    )
    mock_order_service.get_order_status_by_id.assert_not_called()

    # --- Step 2: User provides ONLY the order ID -> Agent provides status ---
    user_input_2 = order_id # Pass the 32-char ID directly
    mock_llm_service.determine_intent.reset_mock()
    # Assume intent is re-checked or agent handles direct ID input
    mock_llm_service.determine_intent.return_value = 'check_order_status'
    mock_order_service.get_order_status_by_id.return_value = sample_order_data_found

    response_data_2 = await conversation_manager.handle_message(user_input_2, test_session_id)

    expected_formatted_details = format_order_details(sample_order_data_found)
    assert response_data_2["response"] == expected_formatted_details
    mock_order_service.get_order_status_by_id.assert_called_once_with(order_id)
    # Check intent determination for the second message
    mock_llm_service.determine_intent.assert_called_once()
    args_step2, kwargs_step2 = mock_llm_service.determine_intent.call_args
    assert kwargs_step2.get('user_input') == user_input_2
    # History should now contain the first interaction
    assert len(kwargs_step2.get('history', [])) == 2 # user_input_1, response_data_1
    assert kwargs_step2['history'][0]['role'] == 'user'
    assert kwargs_step2['history'][0]['parts'][0] == user_input_1
    assert kwargs_step2['history'][1]['role'] == 'model'
    assert kwargs_step2['history'][1]['parts'][0] == response_data_1["response"]


@pytest.mark.asyncio
async def test_order_status_agent_not_found(
    conversation_manager: ConversationManager,
    mock_llm_service: AsyncMock,
    mock_order_service: AsyncMock,
    test_session_id: str
):
    """Test OrderStatusAgent response when order is not found (simulating 2 steps)."""
    non_existent_order_id = "11111111111111111111111111111111" # Example 32-char non-existent ID
    assert len(non_existent_order_id) == 32

    # --- Step 1: User asks to check status -> Agent asks for ID ---
    user_input_1 = "where is my order"
    mock_llm_service.determine_intent.reset_mock()
    mock_order_service.get_order_status_by_id.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status'

    response_data_1 = await conversation_manager.handle_message(user_input_1, test_session_id)

    assert "Please provide the 32-character alphanumeric order ID" in response_data_1["response"]
    mock_llm_service.determine_intent.assert_called_once_with(
        user_input=user_input_1,
        available_intents=conversation_manager.available_intents,
        history=[]
    )
    mock_order_service.get_order_status_by_id.assert_not_called()

    # --- Step 2: User provides ONLY the (non-existent) order ID -> Agent says not found ---
    user_input_2 = non_existent_order_id
    mock_llm_service.determine_intent.reset_mock()
    mock_llm_service.determine_intent.return_value = 'check_order_status'
    mock_order_service.get_order_status_by_id.return_value = None # Mock service returns None

    response_data_2 = await conversation_manager.handle_message(user_input_2, test_session_id)

    expected_not_found_msg = f"Sorry, I couldn't find any order with the ID '{non_existent_order_id}'. Please double-check the ID."
    # Use 'in' for flexibility in case of minor wording changes
    assert expected_not_found_msg in response_data_2["response"]
    mock_order_service.get_order_status_by_id.assert_called_once_with(non_existent_order_id)
    # Check intent determination for the second message
    mock_llm_service.determine_intent.assert_called_once()
    args_step2, kwargs_step2 = mock_llm_service.determine_intent.call_args
    assert kwargs_step2.get('user_input') == user_input_2
    assert len(kwargs_step2.get('history', [])) == 2 # Check history includes first turn

# --- Helper Function Tests (Synchronous) ---

def test_format_order_details_delivered(sample_order_data_found):
    """Test formatting for a delivered order using fixture data."""
    # Assumes sample_order_data_found has realistic datetime objects and 32-char ID
    order_id = sample_order_data_found.order_id
    assert isinstance(order_id, str) and len(order_id) == 32

    formatted_details = format_order_details(sample_order_data_found)

    assert f"Order ID: {order_id}" in formatted_details
    assert ORDER_STATUS_DESCRIPTIONS.get('delivered', 'delivered') in formatted_details
    # Check for specific date formats if the fixture guarantees them
    if sample_order_data_found.order_purchase_timestamp:
         assert f"Purchased on: {sample_order_data_found.order_purchase_timestamp:%Y-%m-%d %H:%M}" in formatted_details
    if sample_order_data_found.order_estimated_delivery_date:
         assert f"Estimated Delivery: {sample_order_data_found.order_estimated_delivery_date:%Y-%m-%d}" in formatted_details
    if sample_order_data_found.order_delivered_customer_date:
         assert f"Delivered on: {sample_order_data_found.order_delivered_customer_date:%Y-%m-%d %H:%M}" in formatted_details

def test_format_order_details_invoiced(sample_order_data_invoiced):
    """Test formatting for an invoiced (not delivered) order using fixture data."""
    # Assumes sample_order_data_invoiced has realistic datetime objects and 32-char ID
    order_id = sample_order_data_invoiced.order_id
    assert isinstance(order_id, str) and len(order_id) == 32

    formatted_details = format_order_details(sample_order_data_invoiced)

    assert f"Order ID: {order_id}" in formatted_details
    assert ORDER_STATUS_DESCRIPTIONS.get('invoiced', 'invoiced') in formatted_details
    if sample_order_data_invoiced.order_purchase_timestamp:
         assert f"Purchased on: {sample_order_data_invoiced.order_purchase_timestamp:%Y-%m-%d %H:%M}" in formatted_details
    if sample_order_data_invoiced.order_estimated_delivery_date:
         assert f"Estimated Delivery: {sample_order_data_invoiced.order_estimated_delivery_date:%Y-%m-%d}" in formatted_details
    # Ensure "Delivered on" is NOT present for non-delivered statuses
    assert "Delivered on:" not in formatted_details

def test_format_order_details_none():
    """Test formatting when None is passed."""
    formatted = format_order_details(None)
    assert formatted == "Order details could not be retrieved."

# --- Direct Agent Unit Test ---
@pytest.mark.asyncio
async def test_order_status_agent_process_direct_id(
    mock_llm_service: AsyncMock,
    mock_order_service: AsyncMock,
    sample_order_data_found: MagicMock,
    test_session_id: str
):
    """Test OrderStatusAgent.process directly when input is just the ID."""
    # Instantiate the agent directly, injecting the mock service
    agent = OrderStatusAgent(llm_service=mock_llm_service, order_service=mock_order_service)
    state = ConversationState(session_id=test_session_id) # Provide state object

    order_id = sample_order_data_found.order_id
    user_input = order_id # Input is just the ID
    assert len(user_input) == 32

    # Configure mock service
    mock_order_service.get_order_status_by_id.reset_mock()
    mock_order_service.get_order_status_by_id.return_value = sample_order_data_found

    # Call the agent's process method
    response = await agent.process(state, user_input)

    # Assertions
    mock_order_service.get_order_status_by_id.assert_called_once_with(order_id)
    expected_formatted_details = format_order_details(sample_order_data_found)
    assert response == expected_formatted_details
    # LLM should not be called by the agent in this specific path (it just gets/formats data)
    mock_llm_service.determine_intent.assert_not_called()
    mock_llm_service.generate_response.assert_not_called()

# --- Test Order ID Extraction Helper ---

# Keep this test if the helper function is still used and relevant
@pytest.mark.parametrize("text, expected_id", [
    ("my order id is abc123def456ghi789jkl012mno345p0 please check", "abc123def456ghi789jkl012mno345p0"),
    ("abc123def456ghi789jkl012mno345p0", "abc123def456ghi789jkl012mno345p0"),
    ("check status for xyz987abc654def321ghi098jkl7650a thanks", "xyz987abc654def321ghi098jkl7650a"),
    ("order 12345", None), # Too short
    ("order abc123def456ghi789jkl012mno345p0!", "abc123def456ghi789jkl012mno345p0"),
    ("no order id here", None),
    ("", None),
    ("The ID is e481f51cbdc54678b7cc49136f2d6af7.", "e481f51cbdc54678b7cc49136f2d6af7"),
    ("53cdb2fc8bc7dce0b6741e2150273451 is the order", "53cdb2fc8bc7dce0b6741e2150273451"),
])
def test_extract_order_id(text, expected_id):
    """Test the order ID extraction helper with various inputs."""
    assert extract_order_id(text) == expected_id