# Testing Strategy

We use `pytest` for automated testing to ensure the reliability and correctness of the chatbot.

## Running Tests

To run all tests, navigate to the project root directory and execute:

```bash
pytest -v
```
## Test Structure (tests/)
```conftest.py: Contains shared fixtures, such as mock services (mock_llm_service, mock_order_service) and sample data (sample_order_data_found). This promotes code reuse in tests.

test_main_flows.py: Contains integration-style tests that verify the end-to-end conversation flow through the ConversationManager and interactions between different components (intent routing, agent processing).

(Recommended) test_agents/: Directory for unit tests focusing on individual agent logic.

(Recommended) test_services/: Directory for unit tests focusing on individual service logic.

(Recommended) test_utils/: Directory for unit tests focusing on helper functions (like test_extract_order_id which you already have, potentially moved here).
```
## Mocking
We heavily rely on mocking (unittest.mock, pytest-mock, pytest-asyncio) to isolate components during testing:
LLM calls are mocked to avoid actual API costs and ensure deterministic intent results.
Service dependencies (like database access in OrderService) are mocked to test agent logic without needing a live database.

## Continuous Integration
Tests are automatically run on every push and pull request using GitHub Actions (see .github/workflows/ci.yml). This helps catch regressions early.