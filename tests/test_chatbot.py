# test_chatbot.py
"""
test_chatbot.py
Unit tests for the E-commerce support chatbot, reflecting graph-based logic.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import json
import pandas as pd # Needed for contact saving test potentially
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock config before importing chatbot if necessary, or ensure config is test-friendly
# Example: Mocking FAQ_CONFIG if it's complex or loaded dynamically
# with patch.dict(sys.modules, {'src.config': MagicMock(FAQ_CONFIG=...)}):
#     from src.chatbot import chat_with_user, detect_intent, reset_state, FAQ_RESPONSES

# Assuming config is loaded statically and correctly
from src.chatbot import chat_with_user, detect_intent, reset_state, FAQ_RESPONSES
from src.config import FAQ_CONFIG # Keep for expected values
from tests.test_helpers import setup_test_credentials, teardown_test_credentials

# --- Constants for Expected Messages (Match chatbot.py/utils.py) ---
ORDER_PROMPT_MSG = ("I'd be happy to help check your order status. "
                    "Could you please provide your order ID or customer ID? "
                    "It should be a 32-character alphanumeric code (letters a-f, numbers 0-9).")
CONTACT_PROMPT_NAME_MSG = "Okay, I can help connect you with a human representative. First, could you please provide your full name?"
CONTACT_PROMPT_EMAIL_MSG = "Next, could you please provide your email address?" # Adjusted based on collect_contact_info
CONTACT_PROMPT_PHONE_MSG = "Great, thank you. Finally, could you please provide your phone number?" # Adjusted
CONTACT_SAVED_MSG_PART = "Thank you! I've recorded your information."
CONTACT_CANCEL_MSG = "Okay, I've canceled the request to speak with a human representative. How else can I help you today?"
ORDER_NOT_FOUND_MSG_PART = "I couldn't find any orders associated with the ID"
# Example formatted success message (needs details matching format_order_details)
ORDER_FOUND_MSG_EXAMPLE = "Okay, here's the information for order #1234567890abcdef1234567890abcdef:\nStatus: **Shipped**\n   - Your order has been shipped and is on its way.\nPurchased on: March 26, 2025\nPayment Approved on: March 26, 2025\nEstimated Delivery: March 30, 2025\nTracking: [Tracking information not available in this demo]\n\nYour order is on its way!"


class TestChatbot(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        setup_test_credentials() # Ensure mock credentials are set
        self.initial_state = reset_state() # Get a fresh state

        # --- Mock the Graph Compilation and Invocation ---
        # We patch 'get_compiled_graph' which is now used by chat_with_user
        patcher_graph = patch("src.chatbot.get_compiled_graph")
        self.addCleanup(patcher_graph.stop)
        self.mock_get_compiled_graph = patcher_graph.start()

        # This mock represents the *compiled graph object*
        self.mock_compiled_graph = MagicMock()
        # Configure the patch to return our mock graph object
        self.mock_get_compiled_graph.return_value = self.mock_compiled_graph

        # --- Mock External Services (if not already handled by graph nodes) ---
        # Mock verify_credentials called within chat_with_user itself
        patcher_creds = patch("src.chatbot.verify_credentials")
        self.addCleanup(patcher_creds.stop)
        self.mock_verify_credentials = patcher_creds.start()
        self.mock_verify_credentials.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}

        # Mock contact saving if testing that specific step
        patcher_save = patch("src.chatbot.contact_service.save_contact_info")
        self.addCleanup(patcher_save.stop)
        self.mock_save_contact_info = patcher_save.start()
        self.mock_save_contact_info.return_value = True # Assume success by default

        # Mock order service if testing lookup results directly (less common now, focus on graph invoke)
        # patcher_order = patch("src.chatbot.order_service")
        # self.addCleanup(patcher_order.stop)
        # self.mock_order_service = patcher_order.start()


    def tearDown(self):
        """Clean up after each test."""
        teardown_test_credentials()

# --- Test detect_intent (Should still work) ---

def test_detect_intent():
    """Test intent detection logic."""
    cases = {
        "What's your return policy?": "return_policy",
        "How long does shipping take?": "shipping_policy",
        "What payment methods do you accept?": "payment_methods",
        "I want to speak to a human": "human_agent",
        "What's the status of my order?": "order_status",
        "track my package": "order_status",
        "Hello there": "greeting",
        "thanks bye": "goodbye",
        "1234567890abcdef1234567890abcdef": None,
        "order 1234567890abcdef1234567890abcdef status": "order_status",
        "order status 1234567890abcdef1234567890abcdef": "order_status",
    }
    for text, expected in cases.items():
        assert detect_intent(text) == expected

    # --- Test Simple FAQ Handling (No Graph Invoke) ---
    def test_faq_response_direct(self):
        """Test that simple FAQs are handled directly without invoking the graph."""
        faq_intent = "return_policy"
        faq_question = "What is your return policy?"
        expected_response = FAQ_RESPONSES.get(faq_intent)
        self.assertIsNotNone(expected_response, f"FAQ response for '{faq_intent}' not found in config.")

        # Act
        result_state = chat_with_user(faq_question, self.initial_state)

        # Assert
        self.assertEqual(len(result_state["messages"]), 2) # User + Assistant
        self.assertEqual(result_state["messages"][0]["content"], faq_question)
        self.assertEqual(result_state["messages"][1]["role"], "assistant")
        self.assertEqual(result_state["messages"][1]["content"], expected_response)
        # Crucially, assert the graph was NOT called
        self.mock_compiled_graph.invoke.assert_not_called()

    def test_greeting_response_direct(self):
        """Test greeting handled directly if configured as simple FAQ."""
        greeting_intent = "greeting"
        greeting_text = "Hello"
        expected_response = FAQ_RESPONSES.get(greeting_intent)
        # Only run this test if greeting is actually in FAQ_RESPONSES
        if expected_response:
            # Act
            result_state = chat_with_user(greeting_text, self.initial_state)
            # Assert
            self.assertEqual(len(result_state["messages"]), 2)
            self.assertEqual(result_state["messages"][1]["content"], expected_response)
            self.mock_compiled_graph.invoke.assert_not_called()
        else:
            self.skipTest("Greeting intent not found in FAQ_RESPONSES for direct handling.")

    # --- Test Graph-Handled Flows (Mocking Graph Invoke) ---

    def test_order_lookup_prompt_via_graph(self):
        """Test asking for order status invokes graph and prompts for ID."""
        user_input = "What's the status of my order?"
        initial_state_with_input = {**self.initial_state, "messages": [{"role": "user", "content": user_input}]}

        # Configure the mock graph's invoke method to return the state *after* the prompt
        mock_return_state = {
            **initial_state_with_input,
            "messages": initial_state_with_input["messages"] + [{"role": "assistant", "content": ORDER_PROMPT_MSG}],
            "order_lookup_attempted": True # Graph node should set this
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        # Act
        result_state = chat_with_user(user_input, self.initial_state)

        # Assert
        self.mock_compiled_graph.invoke.assert_called_once()
        # Check the state passed to invoke
        call_args, _ = self.mock_compiled_graph.invoke.call_args
        state_passed_to_graph = call_args[0]
        self.assertEqual(state_passed_to_graph["messages"][-1]["content"], user_input)

        # Check the final state returned by chat_with_user
        self.assertEqual(len(result_state["messages"]), 2)
        self.assertEqual(result_state["messages"][1]["role"], "assistant")
        self.assertEqual(result_state["messages"][1]["content"], ORDER_PROMPT_MSG)
        self.assertTrue(result_state["order_lookup_attempted"])

    def test_order_lookup_success_via_graph(self):
        """Test providing an ID invokes graph and returns formatted success."""
        user_input = "My order ID is 1234567890abcdef1234567890abcdef"
        initial_state_with_input = {**self.initial_state, "messages": [{"role": "user", "content": user_input}]}

        # Configure mock graph invoke to return state after successful lookup
        # Note: The response content should match format_order_details output
        mock_return_state = {
            **initial_state_with_input,
            "messages": initial_state_with_input["messages"] + [{"role": "assistant", "content": ORDER_FOUND_MSG_EXAMPLE}],
            "order_lookup_attempted": True,
            "current_order_id": "1234567890abcdef1234567890abcdef"
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        # Act
        result_state = chat_with_user(user_input, self.initial_state)

        # Assert
        self.mock_compiled_graph.invoke.assert_called_once()
        self.assertEqual(len(result_state["messages"]), 2)
        self.assertEqual(result_state["messages"][1]["role"], "assistant")
        # Use assertIn for flexibility if exact formatting varies slightly
        self.assertIn("Okay, here's the information for order #1234567890abcdef1234567890abcdef", result_state["messages"][1]["content"])
        self.assertIn("Status: **Shipped**", result_state["messages"][1]["content"])
        self.assertTrue(result_state["order_lookup_attempted"])
        self.assertEqual(result_state["current_order_id"], "1234567890abcdef1234567890abcdef")

    def test_order_lookup_fail_via_graph(self):
        """Test providing an invalid/unknown ID invokes graph and returns 'not found'."""
        user_input = "Check ID 00000000000000000000000000000000"
        initial_state_with_input = {**self.initial_state, "messages": [{"role": "user", "content": user_input}]}

        # Configure mock graph invoke for failed lookup
        mock_return_state = {
            **initial_state_with_input,
            "messages": initial_state_with_input["messages"] + [{"role": "assistant", "content": f"{ORDER_NOT_FOUND_MSG_PART} '00000000000000000000000000000000'. Please double-check the ID..."}],
            "order_lookup_attempted": True,
            "current_order_id": None # Should not be set on failure
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        # Act
        result_state = chat_with_user(user_input, self.initial_state)

        # Assert
        self.mock_compiled_graph.invoke.assert_called_once()
        self.assertEqual(len(result_state["messages"]), 2)
        self.assertIn(ORDER_NOT_FOUND_MSG_PART, result_state["messages"][1]["content"])
        self.assertTrue(result_state["order_lookup_attempted"])
        self.assertIsNone(result_state["current_order_id"])

    def test_contact_flow_start_via_graph(self):
        """Test asking for human invokes graph and starts contact collection."""
        user_input = "I need to talk to an agent"
        initial_state_with_input = {**self.initial_state, "messages": [{"role": "user", "content": user_input}]}

        # Configure mock graph invoke for starting contact flow
        mock_return_state = {
            **initial_state_with_input,
            "messages": initial_state_with_input["messages"] + [{"role": "assistant", "content": CONTACT_PROMPT_NAME_MSG}],
            "needs_human_agent": True,
            "contact_step": 1
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        # Act
        result_state = chat_with_user(user_input, self.initial_state)

        # Assert
        self.mock_compiled_graph.invoke.assert_called_once()
        self.assertEqual(len(result_state["messages"]), 2)
        self.assertEqual(result_state["messages"][1]["content"], CONTACT_PROMPT_NAME_MSG)
        self.assertTrue(result_state["needs_human_agent"])
        self.assertFalse(result_state["contact_info_collected"])
        self.assertEqual(result_state["contact_step"], 1)

    def test_contact_flow_provide_name_via_graph(self):
        """Test providing name during contact flow invokes graph and asks for email."""
        user_input = "My name is Test User"
        # State *before* this input
        state_before = {
            **self.initial_state,
            "messages": [
                {"role": "user", "content": "I need to talk to an agent"},
                {"role": "assistant", "content": CONTACT_PROMPT_NAME_MSG}
            ],
            "needs_human_agent": True,
            "contact_step": 1
        }
        state_with_input = {
            **state_before,
            "messages": state_before["messages"] + [{"role": "user", "content": user_input}]
        }

        # Configure mock graph invoke for the next step (asking for email)
        mock_return_state = {
            **state_with_input,
            "messages": state_with_input["messages"] + [{"role": "assistant", "content": f"Thank you, {user_input}. {CONTACT_PROMPT_EMAIL_MSG}"}],
            "needs_human_agent": True,
            "contact_step": 2,
            "customer_name": "Test User"
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        # Act
        result_state = chat_with_user(user_input, state_before) # Pass state *before* user input

        # Assert
        self.mock_compiled_graph.invoke.assert_called_once()
        # Check state passed to graph included the new user message
        call_args, _ = self.mock_compiled_graph.invoke.call_args
        state_passed_to_graph = call_args[0]
        self.assertEqual(state_passed_to_graph["messages"][-1]["content"], user_input)
        self.assertEqual(state_passed_to_graph["contact_step"], 1) # Step before processing

        # Check final state
        self.assertEqual(len(result_state["messages"]), 4) # agent, name_prompt, user_name, email_prompt
        self.assertIn(CONTACT_PROMPT_EMAIL_MSG, result_state["messages"][-1]["content"])
        self.assertTrue(result_state["needs_human_agent"])
        self.assertEqual(result_state["contact_step"], 2)
        self.assertEqual(result_state["customer_name"], "Test User")

    def test_contact_flow_provide_email_via_graph(self):
        """Test providing email during contact flow invokes graph and asks for phone."""
        user_input = "test@example.com"
        state_before = {
            **self.initial_state,
            "messages": [
                {"role": "user", "content": "I need agent"},
                {"role": "assistant", "content": CONTACT_PROMPT_NAME_MSG},
                {"role": "user", "content": "Test User"},
                {"role": "assistant", "content": f"Thank you, Test User. {CONTACT_PROMPT_EMAIL_MSG}"}
            ],
            "needs_human_agent": True,
            "contact_step": 2,
            "customer_name": "Test User"
        }
        state_with_input = {
            **state_before,
            "messages": state_before["messages"] + [{"role": "user", "content": user_input}]
        }
        mock_return_state = {
            **state_with_input,
            "messages": state_with_input["messages"] + [{"role": "assistant", "content": CONTACT_PROMPT_PHONE_MSG}],
            "needs_human_agent": True,
            "contact_step": 3,
            "customer_name": "Test User",
            "customer_email": "test@example.com"
        }
        self.mock_compiled_graph.invoke.return_value = mock_return_state

        result_state = chat_with_user(user_input, state_before)

        self.mock_compiled_graph.invoke.assert_called_once()
        self.assertEqual(len(result_state["messages"]), 6)
        self.assertEqual(result_state["messages"][-1]["content"], CONTACT_PROMPT_PHONE_MSG)
        self.assertEqual(result_state["contact_step"], 3)
        self.assertEqual(result_state["customer_email"], "test@example.com")

    def test_contact_flow_provide_phone_via_graph_and_save(self):
      """Test providing phone completes flow, invokes graph, and confirms (state check).""" # Modified docstring
      user_input = "123-456-7890"
      state_before = {
          **self.initial_state,
          "messages": [
              # Simplified previous messages for clarity
              {"role": "user", "content": "I need agent"},
              {"role": "assistant", "content": CONTACT_PROMPT_NAME_MSG},
              {"role": "user", "content": "Test User"},
              {"role": "assistant", "content": f"Thank you, Test User. {CONTACT_PROMPT_EMAIL_MSG}"},
              {"role": "user", "content": "test@example.com"},
              {"role": "assistant", "content": CONTACT_PROMPT_PHONE_MSG}
          ],
          "needs_human_agent": True,
          "contact_step": 3,
          "customer_name": "Test User",
          "customer_email": "test@example.com"
      }
      state_with_input = {
          **state_before,
          "messages": state_before["messages"] + [{"role": "user", "content": user_input}]
      }
      # Mock graph return state *after* saving and confirmation
      mock_return_state = {
          **state_with_input,
          "messages": state_with_input["messages"] + [{"role": "assistant", "content": f"{CONTACT_SAVED_MSG_PART} A customer service representative will contact you soon..."}],
          "needs_human_agent": False, # Set to False after collection completes
          "contact_info_collected": True,
          "contact_step": 4, # Final step
          "customer_name": "Test User",
          "customer_email": "test@example.com",
          "customer_phone": "123-456-7890"
      }
      self.mock_compiled_graph.invoke.return_value = mock_return_state
  
      # Act
      result_state = chat_with_user(user_input, state_before)
  
      # Assert Graph Invoke
      self.mock_compiled_graph.invoke.assert_called_once()
      call_args, _ = self.mock_compiled_graph.invoke.call_args
      state_passed_to_graph = call_args[0]
      self.assertEqual(state_passed_to_graph["messages"][-1]["content"], user_input) # Check input was passed
  
      # Assert Save Called - REMOVED because invoke is mocked
      # self.mock_save_contact_info.assert_called_once_with("Test User", "test@example.com", "123-456-7890")
  
      # Assert Final State (based on mocked return)
      self.assertEqual(len(result_state["messages"]), 8) # 6 initial + 1 user + 1 final assistant
      self.assertIn(CONTACT_SAVED_MSG_PART, result_state["messages"][-1]["content"])
      self.assertTrue(result_state["contact_info_collected"])
      self.assertFalse(result_state["needs_human_agent"]) # Check it's reset
      self.assertEqual(result_state["contact_step"], 4)
      self.assertEqual(result_state["customer_phone"], "123-456-7890")

    def test_contact_flow_cancel_direct(self):
      """Test cancelling contact flow is handled directly."""
      user_input = "cancel"
      state_before = {
          **self.initial_state,
          "messages": [
              {"role": "user", "content": "I need agent"},
              {"role": "assistant", "content": CONTACT_PROMPT_NAME_MSG}
          ],
          "needs_human_agent": True,
          "contact_step": 1
      }
  
      # Act: Cancellation is handled *before* graph invoke in chat_with_user
      result_state = chat_with_user(user_input, state_before)
  
      # Assert
      self.mock_compiled_graph.invoke.assert_not_called() # Should not invoke graph
  
      # --- CORRECTED Assertion ---
      # Expected messages:
      # 1. user: I need agent
      # 2. assistant: ...provide name?
      # 3. user: cancel  (added by chat_with_user)
      # 4. assistant: Okay, I've canceled... (added by cancellation logic in chat_with_user)
      self.assertEqual(len(result_state["messages"]), 4)
      # --- END CORRECTED Assertion ---
  
      # Check the content of the last message
      expected_cancel_msg = "Okay, I've canceled the request to speak with a human representative. How else can I help you today?"
      self.assertEqual(result_state["messages"][-1]["role"], "assistant")
      self.assertEqual(result_state["messages"][-1]["content"], expected_cancel_msg)
  
      # Check state reset
      self.assertFalse(result_state["needs_human_agent"])
      self.assertFalse(result_state["contact_info_collected"])
      self.assertEqual(result_state["contact_step"], 0)
      self.assertIsNone(result_state["customer_name"]) # Check partial info cleared
      self.assertIsNone(result_state["customer_email"])
      self.assertIsNone(result_state["customer_phone"])

    def test_graph_invoke_error(self):
      """Test fallback response when graph invoke raises an error."""
      # Use an input guaranteed not to match simple FAQs (adjust if needed based on your FAQ_CONFIG)
      user_input = "Tell me about the general workflow for returns"
      # Ensure this input does NOT trigger direct FAQ handling in chat_with_user
  
      # Configure mock graph invoke to raise an exception
      self.mock_compiled_graph.invoke.side_effect = Exception("Graph execution failed!")
  
      # Act: Call chat_with_user with the initial state and the new input.
      # chat_with_user will add the user_input to the messages list internally.
      result_state = chat_with_user(user_input, self.initial_state)
  
      # Assert: Check that invoke was called (even though it raised an error)
      self.mock_compiled_graph.invoke.assert_called_once()
  
      # Assert: Check that the fallback error message was added
      self.assertEqual(len(result_state["messages"]), 2) # Initial state is empty, add user + fallback assistant
      self.assertEqual(result_state["messages"][0]["role"], "user")
      self.assertEqual(result_state["messages"][0]["content"], user_input)
      self.assertEqual(result_state["messages"][1]["role"], "assistant")
      expected_fallback = ("I seem to be having trouble handling that request right now. "
                           "You could try asking differently, or ask about our return policy, "
                           "shipping options, or request to speak with a human representative.")
      self.assertEqual(result_state["messages"][1]["content"], expected_fallback)

    # --- Test reset_state (Should still work) ---
    def test_reset_state(self):
        """Test resetting the chatbot state."""
        state = reset_state()
        self.assertIsInstance(state, dict)
        self.assertEqual(state["messages"], [])
        self.assertFalse(state["order_lookup_attempted"])
        self.assertIsNone(state["current_order_id"])
        self.assertFalse(state["needs_human_agent"])
        self.assertFalse(state["contact_info_collected"])
        self.assertEqual(state["contact_step"], 0)
        self.assertIsNone(state["customer_name"])
        self.assertIsNone(state["customer_email"])
        self.assertIsNone(state["customer_phone"])
        self.assertIsInstance(state["session_id"], str)
        self.assertGreater(len(state["session_id"]), 10) # Basic check for timestamp format


if __name__ == "__main__":
    unittest.main()
