# test_chatbot.py
import os
import sys
import unittest
from unittest.mock import patch, MagicMock, mock_open
import pandas as pd
from typing import Dict, Any
from datetime import datetime
import re

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
        # Mock LLM service to prevent actual API calls
        self.llm_patcher = patch('src.chatbot.llm_service.generate_response')
        self.mock_llm = self.llm_patcher.start()
        self.mock_llm.return_value = FAQ_CONFIG["responses"]["greeting"]

    def tearDown(self):
        """Clean up after each test."""
        self.llm_patcher.stop()

    def test_detect_intent(self):
        """Test the intent detection function."""
        self.assertEqual(detect_intent("What's your return policy?"), "return_policy")
        self.assertEqual(detect_intent("How long does shipping take?"), "shipping_policy")
        self.assertEqual(detect_intent("What payment methods do you accept?"), "payment_methods")
        self.assertEqual(detect_intent("I want to speak to a human"), "human_agent")
        self.assertEqual(detect_intent("What's the status of my order?"), "order_status")
        self.assertEqual(detect_intent("Hello, how are you today?"), "greeting")

    @patch('src.chatbot.detect_intent')
    def test_chat_with_user_greeting(self, mock_detect_intent):
        """Test the chatbot's greeting."""
        # Mock the intent detection to return "greeting"
        mock_detect_intent.return_value = "greeting"
        
        state = chat_with_user("Hello", self.initial_state)
        self.assertEqual(len(state["messages"]), 2)  # One user message, one bot response
        self.assertEqual(state["messages"][0]["role"], "user")
        self.assertEqual(state["messages"][0]["content"], "Hello")
        self.assertEqual(state["messages"][1]["role"], "assistant")
        self.assertIn("Welcome to our e-commerce support", 
                     state["messages"][1]["content"])

    def test_chat_with_user_faq(self):
        """Test the chatbot's response to FAQ questions."""
        # Test return policy
        with patch('src.chatbot.detect_intent', return_value="return_policy"):
            state = chat_with_user("What's your return policy?", self.initial_state)
            self.assertIn("30 days", state["messages"][-1]["content"])

        # Test shipping policy
        with patch('src.chatbot.detect_intent', return_value="shipping_policy"):
            state = chat_with_user("How long does shipping take?", self.initial_state)
            self.assertIn("Standard shipping (5-7 business days)", 
                         state["messages"][-1]["content"])

        # Test payment methods
        with patch('src.chatbot.detect_intent', return_value="payment_methods"):
            state = chat_with_user("What payment methods do you accept?", self.initial_state)
            self.assertIn("credit cards", state["messages"][-1]["content"].lower())

    @patch('src.chatbot.order_service.lookup_order_by_id')    
    def test_chat_with_user_order_lookup_success(self, mock_lookup):
        """Test successful order lookup."""
        mock_lookup.return_value = ("delivered", {
            'purchase_date': 'January 01, 2023',
            'approved_date': 'January 01, 2023',
            'estimated_delivery': 'January 10, 2023',
            'actual_delivery': 'January 05, 2023'
        })

        state = chat_with_user("What's the status of my order TEST123?", self.initial_state)
        self.assertIn("order ID or customer ID", state["messages"][-1]["content"])
        self.assertFalse(state["order_lookup_attempted"])

    @patch('src.chatbot.llm_service.generate_response')
    def test_chat_with_user_continue_conversation(self, mock_generate):
        """Test conversation continuation."""
        # Set up mock response
        mock_generate.return_value = FAQ_CONFIG["responses"]["greeting"]
        
        # Create a clean state for this test
        test_state = reset_state()
        state = chat_with_user("Hello, can you help me?", test_state)
        
        # Verify the state
        self.assertEqual(len(state["messages"]), 2)  # One user message, one bot response
        self.assertEqual(state["messages"][0]["role"], "user")
        self.assertEqual(state["messages"][0]["content"], "Hello, can you help me?")
        self.assertEqual(state["messages"][1]["role"], "assistant")
        self.assertIn("Welcome to", state["messages"][1]["content"])

    def test_chat_with_user_reset_state(self):
        """Test that the reset_state function returns a valid initial state."""
        initial_state = reset_state()
        expected_keys = {
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
            "session_id": "",  # Will be checked separately
            "feedback": None,
            "type": "messages"
        }
        
        # Check all expected keys exist with correct default values
        for key, expected_value in expected_keys.items():
            self.assertIn(key, initial_state)
            if key != "session_id":  # Skip session_id as it's dynamic
                self.assertEqual(initial_state[key], expected_value)
        
        # Verify session_id is a string with the correct format
        self.assertIsInstance(initial_state["session_id"], str)
        self.assertRegex(initial_state["session_id"], r"^\d{14}$")
