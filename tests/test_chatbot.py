# test_chatbot.py
import os
import sys
import unittest
import pandas as pd
from unittest.mock import patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.chatbot import chat_with_user
from src.utils import get_order_status, format_order_details, load_order_data

class TestChatbot(unittest.TestCase):
    """Test cases for the e-commerce chatbot."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a sample orders dataframe for testing
        self.sample_orders = pd.DataFrame({
            'order_id': ['TEST123', 'TEST456'],
            'customer_id': ['CUST1', 'CUST2'],
            'order_status': ['delivered', 'processing'],
            'order_purchase_timestamp': ['2023-01-01 10:00:00', '2023-02-01 11:00:00'],
            'order_approved_at': ['2023-01-01 10:30:00', '2023-02-01 11:30:00'],
            'order_delivered_carrier_date': ['2023-01-03 09:00:00', None],
            'order_delivered_customer_date': ['2023-01-05 14:00:00', None],
            'order_estimated_delivery_date': ['2023-01-10 00:00:00', '2023-02-15 00:00:00']
        })

    @patch('src.utils.load_order_data')
    def test_get_order_status(self, mock_load_data):
        """Test retrieving order status."""
        mock_load_data.return_value = self.sample_orders

        # Test existing order
        status, details = get_order_status(self.sample_orders, 'TEST123')
        self.assertEqual(status, 'delivered')
        self.assertIsNotNone(details)

        # Test non-existent order
        status, details = get_order_status(self.sample_orders, 'NONEXISTENT')
        self.assertIsNone(status)
        self.assertIsNone(details)

    def test_format_order_details(self):
        """Test formatting order details into a response."""
        order_id = "TEST123"
        status = "delivered"
        details = {
            'purchase_date': 'January 01, 2023',
            'approved_date': 'January 01, 2023',
            'estimated_delivery': 'January 10, 2023',
            'actual_delivery': 'January 05, 2023'
        }

        response = format_order_details(order_id, status, details)

        # Check that the response contains key information
        self.assertIn(order_id, response)
        self.assertIn(status, response)
        self.assertIn(details['purchase_date'], response)
        self.assertIn(details['actual_delivery'], response)

    @patch('src.chatbot.create_chatbot')
    def test_chat_with_user_order_status(self, mock_create_chatbot):
        """Test chatbot response to order status inquiry."""
        # Mock the chatbot to return a predefined state
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "What's the status of my order TEST123?"},
                {"role": "assistant", "content": "Your order TEST123 is delivered."}
            ],
            "order_lookup_attempted": True,
            "current_order_id": "TEST123",
            "needs_human_agent": False,
            "contact_info_collected": False,
            "customer_name": None,
            "customer_email": None
        }
        mock_create_chatbot.return_value = mock_graph

        # Test the chat function
        result = chat_with_user("What's the status of my order TEST123?")

        # Verify the result
        self.assertTrue(result["order_lookup_attempted"])
        self.assertEqual(result["current_order_id"], "TEST123")
        self.assertEqual(len(result["messages"]), 2)
        self.assertIn("delivered", result["messages"][1]["content"])

    @patch('src.chatbot.create_chatbot')
    def test_chat_with_user_human_request(self, mock_create_chatbot):
        """Test chatbot response to human agent request."""
        # Mock the chatbot to return a predefined state
        mock_graph = MagicMock()
        mock_graph.invoke.return_value = {
            "messages": [
                {"role": "user", "content": "I want to speak to a human representative"},
                {"role": "assistant", "content": "I understand you'd like to speak with a human representative. I'll help connect you."}
            ],
            "order_lookup_attempted": False,
            "current_order_id": None,
            "needs_human_agent": True,
            "contact_info_collected": False,
            "customer_name": None,
            "customer_email": None
        }
        mock_create_chatbot.return_value = mock_graph

        # Test the chat function
        result = chat_with_user("I want to speak to a human representative")

        # Verify the result
        self.assertTrue(result["needs_human_agent"])
        self.assertFalse(result["contact_info_collected"])
        self.assertIn("human representative", result["messages"][1]["content"])

    @patch('src.utils.load_order_data')
    def test_load_order_data(self, mock_pd_read_csv):
        """Test loading order data."""
        # Mock pandas read_csv
        mock_pd_read_csv.return_value = self.sample_orders

        # Test with cache enabled
        result = load_order_data(use_cache=True)
        self.assertEqual(len(result), len(self.sample_orders))

        # Test with cache disabled
        result = load_order_data(use_cache=False)
        self.assertEqual(len(result), len(self.sample_orders))

if __name__ == '__main__':
    unittest.main()