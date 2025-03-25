import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import (
    chat_with_user,
    detect_intent,
    FAQ_RESPONSES,
    order_service,
    contact_service,
    reset_state
)

from src.config import FAQ_CONFIG

class TestChatbot(unittest.TestCase):
    def setUp(self):
        """Set up test environment before each test."""
        self.initial_state = reset_state()
        
        # Set up mock chatbot
        self.chatbot_patcher = patch('src.chatbot.create_chatbot')
        self.mock_chatbot_creator = self.chatbot_patcher.start()
        self.mock_chatbot = MagicMock()
        self.mock_chatbot_creator.return_value = self.mock_chatbot
        
        # Create base mock state for reuse
        self.base_mock_state = {
            "messages": [],
            "order_lookup_attempted": False,
            "current_order_id": None,
            "needs_human_agent": False,
            "contact_info_collected": False,
            "customer_name": None,
            "customer_email": None,
            "customer_phone": None,
            "contact_step": 0,
            "chat_history": [],
            "type": "messages"
        }

    def tearDown(self):
        """Clean up after each test."""
        self.chatbot_patcher.stop()

    def _get_mock_state_with_messages(self, user_input: str, bot_response: str) -> dict:
        """Helper method to create mock state with messages"""
        state = self.base_mock_state.copy()
        state["messages"] = [
            {"role": "user", "content": user_input},
            {"role": "assistant", "content": bot_response}
        ]
        return state

    def test_detect_intent(self):
        """Test the intent detection function."""
        test_cases = {
            "What's your return policy?": "return_policy",
            "How long does shipping take?": "shipping_policy",
            "What payment methods do you accept?": "payment_methods",
            "I want to speak to a human": "human_agent",
            "What's the status of my order?": "order_status",
            "Hello, how are you today?": "greeting"
        }
        
        for input_text, expected_intent in test_cases.items():
            with self.subTest(input_text=input_text):
                self.assertEqual(detect_intent(input_text), expected_intent)

    @patch('src.chatbot.detect_intent')
    def test_chat_with_user_greeting(self, mock_detect_intent):
        """Test the chatbot's greeting."""
        mock_detect_intent.return_value = "greeting"
        mock_response = FAQ_CONFIG["responses"]["greeting"]
        
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "Hello", mock_response
        )
        
        state = chat_with_user("Hello", self.initial_state)
        
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["messages"][0], {"role": "user", "content": "Hello"})
        self.assertEqual(state["messages"][1], {"role": "assistant", "content": mock_response})

    def test_chat_with_user_faq(self):
        """Test the chatbot's response to FAQ questions."""
        faq_test_cases = {
            "return_policy": ("What's your return policy?", "30 days"),
            "shipping_policy": ("How long does shipping take?", "Standard shipping (5-7 business days)"),
            "payment_methods": ("What payment methods do you accept?", "credit cards")
        }
        
        for intent, (question, expected_text) in faq_test_cases.items():
            with self.subTest(intent=intent):
                with patch('src.chatbot.detect_intent', return_value=intent):
                    mock_response = FAQ_CONFIG["responses"][intent]
                    self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
                        question, mock_response
                    )
                    
                    state = chat_with_user(question, self.initial_state)
                    self.assertIn(expected_text, state["messages"][-1]["content"].lower())

    @patch('src.chatbot.order_service.lookup_order_by_id')    
    def test_chat_with_user_order_lookup(self, mock_lookup):
        """Test order lookup functionality."""
        # Test successful lookup
        mock_lookup.return_value = ("delivered", {
            'purchase_date': '2025-01-01',
            'approved_date': '2025-01-01',
            'estimated_delivery': '2025-01-10',
            'actual_delivery': '2025-01-05'
        })
        
        mock_response = "To check your order status, I'll need your order ID or customer ID."
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "What's the status of my order TEST123?", mock_response
        )
        
        state = chat_with_user("What's the status of my order TEST123?", self.initial_state)
        self.assertIn("order ID or customer ID", state["messages"][-1]["content"])
        self.assertFalse(state["order_lookup_attempted"])

    @patch('src.chatbot.contact_service.save_contact_info')
    def test_chat_with_user_contact_flow(self, mock_save):
        """Test contact information collection flow."""
        mock_save.return_value = True
        contact_flow = [
            ("I want to speak to a human", "Could you please provide your name?", 1),
            ("John Doe", "Could you please provide your email address?", 2),
            ("john@example.com", "Could you please provide your phone number?", 3),
            ("555-123-4567", "Thank you for providing your information", 4)
        ]
        
        state = self.initial_state
        for user_input, expected_response, step in contact_flow:
            self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
                user_input, expected_response
            )
            state = chat_with_user(user_input, state)
            self.assertIn(expected_response, state["messages"][-1]["content"])
            self.assertEqual(state.get("contact_step", 0), step)

    def test_reset_state(self):
        """Test reset_state functionality."""
        state = reset_state()
        
        # Test all required keys exist with correct default values
        self.assertIsInstance(state, dict)
        self.assertEqual(state["messages"], [])
        self.assertFalse(state["order_lookup_attempted"])
        self.assertIsNone(state["current_order_id"])
        self.assertFalse(state["needs_human_agent"])
        self.assertEqual(state["contact_step"], 0)
        
        # Verify session_id format
        self.assertIsInstance(state["session_id"], str)
        self.assertRegex(state["session_id"], r"^\d{14}$")

if __name__ == '__main__':
    unittest.main()
