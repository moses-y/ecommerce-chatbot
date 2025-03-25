# test_chatbot.py
import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from datetime import datetime
import json

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
        # Create mock credentials file
        self.credentials_dir = os.path.join(os.getcwd(), "test_credentials")
        os.makedirs(self.credentials_dir, exist_ok=True)
        
        # Create a mock credentials file
        self.mock_credentials = {
            "type": "service_account",
            "project_id": "test-project",
            "private_key_id": "test-key-id",
            "private_key": "test-private-key",
            "client_email": "test@test-project.iam.gserviceaccount.com",
            "client_id": "test-client-id",
            "auth_uri": "https://accounts.google.com/o/oauth2/auth",
            "token_uri": "https://oauth2.googleapis.com/token",
            "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
            "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test%40test-project.iam.gserviceaccount.com"
        }
        
        self.credentials_path = os.path.join(self.credentials_dir, "google_credentials.json")
        with open(self.credentials_path, 'w') as f:
            json.dump(self.mock_credentials, f)

        # Set up environment variables
        os.environ["GOOGLE_API_KEY"] = "test_api_key"
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.credentials_path
        os.environ["HUGGINGFACE_TOKEN"] = "test_token"
        
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
            "type": "messages",
            "session_id": "20250325173227"
        }

        # Define standard responses
        self.contact_responses = {
            "name_request": "I'll connect you with a human representative. Could you please provide your name?",
            "email_request": "Could you please provide your email address?",
            "phone_request": "Thank you. Finally, could you please provide your phone number?",
            "thank_you": "Thank you for providing your information. A customer service representative will contact you soon at {email} or {phone}. Is there anything else you'd like to add before I submit your request?"
        }

    def tearDown(self):
        """Clean up after each test."""
        self.chatbot_patcher.stop()
        
        # Clean up credentials file and directory
        if os.path.exists(self.credentials_path):
            os.remove(self.credentials_path)
        if os.path.exists(self.credentials_dir):
            os.rmdir(self.credentials_dir)
            
        # Clean up environment variables
        for key in ["GOOGLE_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS", "HUGGINGFACE_TOKEN"]:
            os.environ.pop(key, None)

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

    @patch('src.chatbot.verify_credentials')
    @patch('src.chatbot.LLMService')
    def test_chat_with_user_greeting(self, mock_llm, mock_verify):
        """Test the chatbot's greeting."""
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        
        initial_response = FAQ_CONFIG["responses"]["greeting"]
        state = self._get_mock_state_with_messages("Hello", initial_response)
        state["messages"] = [{"role": "user", "content": "Hello"}]  # Only include user message initially
        
        self.mock_chatbot.invoke.return_value = state
        state = chat_with_user("Hello", self.initial_state)
        
        self.assertEqual(len(state["messages"]), 2)
        self.assertEqual(state["messages"][0], {"role": "user", "content": "Hello"})
        self.assertIn("Welcome to our e-commerce support", state["messages"][1]["content"])

    @patch('src.chatbot.verify_credentials')
    def test_chat_with_user_faq(self, mock_verify):
        """Test the chatbot's response to FAQ questions."""
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}

        state = chat_with_user("What's your return policy?", self.initial_state)
        self.assertIn("return policy", state["messages"][1]["content"].lower())
        self.assertIn("30 days", state["messages"][1]["content"].lower())

    @patch('src.chatbot.verify_credentials')
    def test_chat_with_user_order_lookup(self, mock_verify):
        """Test order lookup functionality."""
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        
        state = chat_with_user("What's the status of my order?", self.initial_state)
        self.assertIn("order ID or customer ID", state["messages"][1]["content"])
        self.assertFalse(state.get("order_lookup_attempted", True))

    @patch('src.chatbot.verify_credentials')
    @patch('src.chatbot.contact_service.save_contact_info')
    def test_chat_with_user_contact_flow(self, mock_save, mock_verify):
        """Test contact information collection flow."""
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        mock_save.return_value = True
        
        # Initial request
        state = chat_with_user("I want to speak to a human", self.initial_state)
        self.assertEqual(state.get("contact_step", 0), 1)
        self.assertIn("Could you please provide your name?", state["messages"][1]["content"])

        # Provide name
        state = chat_with_user("John Doe", state)
        self.assertEqual(state.get("contact_step", 0), 2)
        self.assertIn("email address", state["messages"][-1]["content"])

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
