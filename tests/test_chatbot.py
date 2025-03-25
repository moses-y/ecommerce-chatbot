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
from tests.test_helpers import setup_test_credentials, teardown_test_credentials

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

        # Define standard responses
        self.contact_responses = {
            "name_request": "I'll connect you with a human representative. Could you please provide your name?",
            "email_request": "Could you please provide your email address?",
            "phone_request": "Thank you. Finally, could you please provide your phone number?",
            "thank_you": "Thank you for providing your information. A customer service representative will contact you soon"
        }

    @classmethod
    def setUpClass(cls):
        """Set up test environment before all tests."""
        setup_test_credentials()

    @classmethod
    def tearDownClass(cls):
        """Clean up test environment after all tests."""
        teardown_test_credentials()
        
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

    def test_chat_with_user_greeting(self):
        """Test the chatbot's greeting."""
        with patch('src.chatbot.detect_intent', return_value="greeting"):
            # Create a response that includes both the user's message and the bot's response
            initial_response = "Hello! I'm your e-commerce assistant. How can I help you today?"
            self.mock_chatbot.invoke.return_value = {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": initial_response}
                ]
            }
            
            state = chat_with_user("Hello", self.initial_state)
            
            # Verify the state contains both messages
            self.assertEqual(len(state["messages"]), 2)
            self.assertEqual(state["messages"][0], {"role": "user", "content": "Hello"})
            self.assertEqual(state["messages"][1], {"role": "assistant", "content": initial_response})

    def test_chat_with_user_faq(self):
        """Test the chatbot's response to FAQ questions."""
        faq_test_cases = {
            "return_policy": {
                "question": "What's your return policy?",
                "response": "Our return policy allows returns within 30 days of purchase. All items must be in original condition with tags attached."
            },
            "shipping_policy": {
                "question": "How long does shipping take?",
                "response": "We offer several shipping options: Standard shipping (5-7 business days): Free for orders over $35"
            },
            "payment_methods": {
                "question": "What payment methods do you accept?",
                "response": "We accept all major credit cards (Visa, MasterCard, American Express, Discover), PayPal, and Apple Pay."
            }
        }
        
        for intent, data in faq_test_cases.items():
            with self.subTest(intent=intent):
                with patch('src.chatbot.detect_intent', return_value=intent):
                    self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
                        data["question"],
                        data["response"]
                    )
                    
                    state = chat_with_user(data["question"], self.initial_state)
                    self.assertEqual(state["messages"][-1]["content"], data["response"])

    @patch('src.chatbot.order_service.lookup_order_by_id')    
    def test_chat_with_user_order_lookup(self, mock_lookup):
        """Test order lookup functionality."""
        # Test initial order status query
        mock_response = "To check your order status, I'll need your order ID or customer ID."
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "What's the status of my order?",
            mock_response
        )
        
        state = chat_with_user("What's the status of my order?", self.initial_state)
        self.assertEqual(state["messages"][-1]["content"], mock_response)
        self.assertFalse(state["order_lookup_attempted"])

    @patch('src.chatbot.contact_service.save_contact_info')
    def test_chat_with_user_contact_flow(self, mock_save):
        """Test contact information collection flow."""
        mock_save.return_value = True
        
        # Initialize state
        state = self.initial_state

        # Step 1: Initial contact request
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "I want to speak to a human",
            self.contact_responses["name_request"]
        )
        state = chat_with_user("I want to speak to a human", state)
        self.assertEqual(state["messages"][-1]["content"], self.contact_responses["name_request"])
        self.assertEqual(state["contact_step"], 1)

        # Step 2: Provide name
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "John Doe",
            self.contact_responses["email_request"]
        )
        state = chat_with_user("John Doe", state)
        self.assertEqual(state["messages"][-1]["content"], self.contact_responses["email_request"])
        self.assertEqual(state["contact_step"], 2)

        # Step 3: Provide email
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "john@example.com",
            self.contact_responses["phone_request"]
        )
        state = chat_with_user("john@example.com", state)
        self.assertEqual(state["messages"][-1]["content"], self.contact_responses["phone_request"])
        self.assertEqual(state["contact_step"], 3)

        # Step 4: Provide phone
        self.mock_chatbot.invoke.return_value = self._get_mock_state_with_messages(
            "555-123-4567",
            self.contact_responses["thank_you"]
        )
        state = chat_with_user("555-123-4567", state)
        self.assertEqual(state["messages"][-1]["content"], self.contact_responses["thank_you"])
        self.assertEqual(state["contact_step"], 4)

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
