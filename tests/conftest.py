# tests/conftest.py (Revised mock_llm_service fixture)
import pytest
import uuid
import regex as re
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

# --- Import necessary classes ---
from src.core.conversation import ConversationManager
# from src.llm.interface import LLMInterface # Interface import not needed if spec is removed
from src.services.order_service import OrderService
from src.services.policy_service import PolicyService
# from src.db.models import Order # Import if needed for spec on other mocks

# --- Basic Fixtures ---

@pytest.fixture
def test_session_id() -> str:
    """Generates a unique session ID for each test function."""
    return f"test_session_{uuid.uuid4()}"

# --- Mock Service Fixtures ---

@pytest.fixture
def mock_llm_service() -> AsyncMock:
    """Provides a mock LLM service."""
    # Create the main mock without spec for now
    mock = AsyncMock()
    # Explicitly create the methods we need to call with 'await' as AsyncMocks
    # Set their default return_value. The test will override this as needed.
    mock.determine_intent = AsyncMock(return_value='unknown')
    mock.generate_response = AsyncMock(return_value="Mock LLM fallback response.")
    return mock

@pytest.fixture
def mock_order_service() -> AsyncMock:
    """Provides a mock OrderService."""
    # Keep spec here as it seemed okay
    mock = AsyncMock(spec=OrderService)
    # Default: return None for not found (Awaitable)
    mock.get_order_status_by_id.return_value = None
    return mock

@pytest.fixture
def mock_policy_service() -> MagicMock: # Policy service might be sync
    """Provides a mock PolicyService."""
    mock = MagicMock(spec=PolicyService)
    mock.get_policy.return_value = "Default mock policy text."
    return mock

# --- Mock Data Fixtures --- (Keep as they were)

@pytest.fixture
def sample_order_data_found() -> MagicMock:
    """Provides a mock Order object representing a found, delivered order."""
    mock_order = MagicMock()
    mock_order.order_id = "ayc123def456ghi789jkl012mno345p7" # 32 chars
    mock_order.order_status = "delivered"
    mock_order.order_purchase_timestamp = datetime(2025, 4, 1, 10, 30)
    mock_order.order_estimated_delivery_date = datetime(2025, 4, 8)
    mock_order.order_delivered_customer_date = datetime(2025, 4, 7, 14, 0)
    return mock_order

@pytest.fixture
def sample_order_data_invoiced() -> MagicMock:
    """Provides a mock Order object representing an invoiced order."""
    mock_order = MagicMock()
    mock_order.order_id = "xyz987abc654def321ghi098jkl7650a" # 32 chars
    mock_order.order_status = "invoiced"
    mock_order.order_purchase_timestamp = datetime(2025, 4, 8, 11, 0)
    mock_order.order_estimated_delivery_date = datetime(2025, 4, 15)
    mock_order.order_delivered_customer_date = None
    return mock_order


# --- Conversation Manager Fixture --- (Keep as it was)

@pytest.fixture
def conversation_manager(
    mock_llm_service: AsyncMock,
    mock_order_service: AsyncMock,
    mock_policy_service: MagicMock
) -> ConversationManager:
    """
    Provides a ConversationManager instance initialized with mock services
    and reloaded agents using those mocks.
    """
    manager = ConversationManager(llm_service=mock_llm_service)
    manager.order_service = mock_order_service
    manager.policy_service = mock_policy_service
    manager.available_services['order_service'] = mock_order_service
    manager.available_services['policy_service'] = mock_policy_service
    manager.agents = manager._load_agents()
    manager.intents = list(manager.agents.keys())
    return manager