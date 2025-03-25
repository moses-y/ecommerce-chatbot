# tests/test_helpers.py
import os
from pathlib import Path
from typing import Optional

def setup_test_credentials(credentials_dir: Optional[str] = None):
    """
    Setup test credentials for unit tests
    Args:
        credentials_dir: Optional directory path where credentials are stored
    """
    # Use provided directory or default to current user's credentials directory
    creds_dir = credentials_dir or str(Path.home() / "credentials")
    
    # Set up mock credentials for testing
    os.environ.update({
        "GOOGLE_API_KEY": "test_google_api_key",
        "HUGGINGFACE_TOKEN": "test_hf_token",
        "GOOGLE_APPLICATION_CREDENTIALS": str(Path(creds_dir) / "google_credentials.json")
    })

def teardown_test_credentials():
    """Remove test credentials from environment"""
    test_keys = [
        "GOOGLE_API_KEY",
        "HUGGINGFACE_TOKEN",
        "GOOGLE_APPLICATION_CREDENTIALS"
    ]
    for key in test_keys:
        os.environ.pop(key, None)