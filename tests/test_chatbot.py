# test_chatbot.py
import os
import sys
import json
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock, mock_open
 
# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import chat_with_user

# Add these test methods to your TestChatbot class

def test_detect_intent(self):
    """Test the intent detection function."""
    from src.chatbot import detect_intent

    # Test return policy intent
    self.assertEqual(detect_intent("What's your return policy?"), "return_policy")

    # Test shipping policy intent
    self.assertEqual(detect_intent("How long does shipping take?"), "shipping_policy")

    # Test human agent intent
    self.assertEqual(detect_intent("I want to speak to a human"), "human_agent")

    # Test order status intent
    self.assertEqual(detect_intent("What's the status of my order?"), "order_status")

    # Test no specific intent
    self.assertIsNone(detect_intent("Hello, how are you today?"))

@patch('src.chatbot.order_service.lookup_order_by_id')
def test_lookup_order_success(self, mock_lookup):
    """Test successful order lookup."""
    from src.chatbot import lookup_order

    # Mock successful order lookup
    mock_lookup.return_value = ("delivered", {
        'purchase_date': 'January 01, 2023',
        'approved_date': 'January 01, 2023',
        'estimated_delivery': 'January 10, 2023',
        'actual_delivery': 'January 05, 2023'
    })

    # Create test state
    state = {
        "messages": [{"role": "user", "content": "What's the status of my order TEST123?"}],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False
    }

    # Call the function
    result = lookup_order(state)

    # Verify results
    self.assertTrue(result["order_lookup_attempted"])
    self.assertEqual(result["current_order_id"], "TEST123")
    self.assertIn("delivered", result["messages"][-1]["content"])

@patch('src.chatbot.order_service.lookup_order_by_id')
def test_lookup_order_not_found(self, mock_lookup):
    """Test order lookup when order is not found."""
    from src.chatbot import lookup_order

    # Mock order not found
    mock_lookup.return_value = (None, None)

    # Create test state
    state = {
        "messages": [{"role": "user", "content": "What's the status of my order NONEXISTENT?"}],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False
    }

    # Call the function
    result = lookup_order(state)

    # Verify results
    self.assertTrue(result["order_lookup_attempted"])
    self.assertIn("couldn't find", result["messages"][-1]["content"].lower())

@patch('src.chatbot.contact_service.save_contact_info')
def test_collect_contact_info(self, mock_save):
    """Test contact information collection."""
    from src.chatbot import collect_contact_info

    # Mock successful save
    mock_save.return_value = True

    # Test initial state
    state = {
        "messages": [{"role": "user", "content": "I want to speak to a human"}],
        "needs_human_agent": True,
        "contact_info_collected": False,
        "contact_step": 0
    }

    # Test step 0 - Ask for name
    result = collect_contact_info(state)
    self.assertEqual(result["contact_step"], 1)
    self.assertIn("provide your name", result["messages"][-1]["content"])

    # Test step 1 - Got name, ask for email
    state = result.copy()
    state["messages"].append({"role": "user", "content": "John Doe"})
    result = collect_contact_info(state)
    self.assertEqual(result["contact_step"], 2)
    self.assertEqual(result["customer_name"], "John Doe")
    self.assertIn("email address", result["messages"][-1]["content"])

    # Test step 2 - Got email, ask for phone
    state = result.copy()
    state["messages"].append({"role": "user", "content": "john@example.com"})
    result = collect_contact_info(state)
    self.assertEqual(result["contact_step"], 3)
    self.assertEqual(result["customer_email"], "john@example.com")
    self.assertIn("phone number", result["messages"][-1]["content"])

    # Test step 3 - Got phone, complete collection
    state = result.copy()
    state["messages"].append({"role": "user", "content": "555-123-4567"})
    result = collect_contact_info(state)
    self.assertEqual(result["contact_step"], 4)
    self.assertEqual(result["customer_phone"], "555-123-4567")
    self.assertTrue(result["contact_info_collected"])
    self.assertIn("representative will contact you", result["messages"][-1]["content"])

@patch('src.chatbot.llm_service.generate_response')
def test_continue_conversation(self, mock_generate):
    """Test conversation continuation."""
    from src.chatbot import continue_conversation

    # Mock LLM response
    mock_generate.return_value = "I'm here to help with your e-commerce questions."

    # Create test state
    state = {
        "messages": [{"role": "user", "content": "Hello, can you help me?"}],
        "order_lookup_attempted": False,
        "current_order_id": None,
        "needs_human_agent": False,
        "contact_info_collected": False
    }

    # Call the function
    result = continue_conversation(state)

    # Verify results
    self.assertEqual(len(result["messages"]), 2)
    self.assertEqual(result["messages"][-1]["role"], "assistant")
    self.assertEqual(result["messages"][-1]["content"], "I'm here to help with your e-commerce questions.")

def test_chat_with_user_faq(self):
    """Test chatbot response to FAQ questions."""
    # Test return policy FAQ
    result = chat_with_user("What's your return policy?")
    self.assertIn("30 days", result["messages"][-1]["content"])

    # Test shipping policy FAQ
    result = chat_with_user("How long does shipping take?")
    self.assertIn("Standard shipping", result["messages"][-1]["content"])

    # Test payment methods FAQ
    result = chat_with_user("What payment methods do you accept?")
    self.assertIn("credit cards", result["messages"][-1]["content"].lower())

@patch('builtins.open', new_callable=mock_open)
@patch('json.dump')
def test_feedback_submission(self, mock_json_dump, mock_file_open):
    """Test feedback submission functionality."""
    from src.chatbot import submit_feedback

    # Create test state
    state = {
        "session_id": "20250323123456",
        "chat_history": [
            ("Hello", "Hi there! How can I help you today?"),
            ("What's your return policy?", "Our return policy allows returns within 30 days...")
        ],
        "feedback": None
    }

    # Submit feedback
    feedback = "Great service, very helpful!"
    result = submit_feedback(feedback, state)

    # Verify results
    self.assertEqual(result["feedback"], feedback)
    mock_file_open.assert_called_once()
    mock_json_dump.assert_called_once()
