# test_chatbot.py
"""
test_chatbot.py
Unit tests for the E-commerce support chatbot.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import json

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.chatbot import chat_with_user, detect_intent, FAQ_RESPONSES, reset_state
from src.config import FAQ_CONFIG
from tests.test_helpers import setup_test_credentials, teardown_test_credentials

class TestChatbot(unittest.TestCase):
    def setUp(self):
        setup_test_credentials()
        self.initial_state = reset_state()
        self.base_state = {
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
            "session_id": "20250325174159",
            "feedback": None,
            "type": "messages"
        }
        patcher = patch("src.chatbot.create_chatbot")
        self.addCleanup(patcher.stop)
        self.mock_create_chatbot = patcher.start()
        self.mock_chatbot = MagicMock()
        self.mock_create_chatbot.return_value = self.mock_chatbot

    def tearDown(self):
        teardown_test_credentials()

    def test_detect_intent(self):
        cases = {
            "What's your return policy?": "return_policy",
            "How long does shipping take?": "shipping_policy",
            "What payment methods do you accept?": "payment_methods",
            "I want to speak to a human": "human_agent",
            "What's the status of my order?": "order_status",
            "Hello": "greeting"
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(detect_intent(text), expected)

    @patch("src.chatbot.verify_credentials")
    def test_greeting_response(self, mock_verify):
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        # Use FAQ response for greeting
        response_text = FAQ_CONFIG["responses"].get("greeting", "Hello! How can I help?")
        state = self.base_state.copy()
        state["messages"] = [{"role": "user", "content": "Hello"}]
        # Mock chatbot invoke returns a state with greeting response.
        self.mock_chatbot.invoke.return_value = {
            **state,
            "messages": [{"role": "user", "content": "Hello"},
                         {"role": "assistant", "content": response_text}]
        }
        result = chat_with_user("Hello", self.initial_state)
        self.assertEqual(len(result["messages"]), 2)
        self.assertEqual(result["messages"][0]["content"], "Hello")
        self.assertIn("Hello", result["messages"][1]["content"])

    @patch("src.chatbot.verify_credentials")
    def test_faq_response(self, mock_verify):
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        state = self.base_state.copy()
        state["messages"] = [{"role": "user", "content": "What is your return policy?"}]
        self.mock_chatbot.invoke.return_value = {
            **state,
            "messages": [{"role": "user", "content": "What is your return policy?"},
                         {"role": "assistant", "content": FAQ_CONFIG["responses"]["return_policy"]}]
        }
        result = chat_with_user("What is your return policy?", self.initial_state)
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("30 days", result["messages"][1]["content"])

    @patch("src.chatbot.verify_credentials")
    def test_order_lookup_prompt(self, mock_verify):
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        state = self.base_state.copy()
        state["messages"] = [{"role": "user", "content": "What's the status of my order?"}]
        self.mock_chatbot.invoke.return_value = {
            **state,
            "messages": [{"role": "user", "content": "What's the status of my order?"},
                         {"role": "assistant", "content": "Please provide your order ID."}]
        }
        result = chat_with_user("What's the status of my order?", self.initial_state)
        self.assertIn("order ID", result["messages"][-1]["content"])

    @patch("src.chatbot.verify_credentials")
    @patch("src.chatbot.contact_service.save_contact_info", return_value=True)
    def test_contact_flow(self, mock_save, mock_verify):
        mock_verify.return_value = {"GOOGLE_API_KEY": True, "GOOGLE_APPLICATION_CREDENTIALS": True}
        # Simulate initiating human agent request.
        state = self.base_state.copy()
        state["messages"] = [{"role": "user", "content": "I want to speak to a human"}]
        state["needs_human_agent"] = True
        state["contact_step"] = 1
        self.mock_chatbot.invoke.return_value = {
            **state,
            "messages": [{"role": "user", "content": "I want to speak to a human"},
                         {"role": "assistant", "content": "Please provide your name."}]
        }
        result = chat_with_user("I want to speak to a human", self.initial_state)
        self.assertEqual(result.get("contact_step"), 1)
        self.assertIn("provide your name", result["messages"][-1]["content"])

    def test_reset_state(self):
        state = reset_state()
        self.assertIsInstance(state, dict)
        self.assertEqual(state["messages"], [])
        self.assertFalse(state["order_lookup_attempted"])
        self.assertIsNone(state["current_order_id"])
        self.assertFalse(state["needs_human_agent"])
        self.assertEqual(state["contact_step"], 0)
        self.assertRegex(state["session_id"], r"^\d{14}$")

if __name__ == "__main__":
    unittest.main()
