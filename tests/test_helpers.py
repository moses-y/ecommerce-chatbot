# tests/test_helpers.py
"""
tests/test_helpers.py
Helper functions to set up and tear down test credentials.
"""
import os
from pathlib import Path
from typing import Optional

def setup_test_credentials(credentials_dir: Optional[str] = None):
    """
    Set up test credentials for unit tests.
    Args:
        credentials_dir: Optional directory path for credentials storage.
    """
    creds_dir = credentials_dir or str(Path.home() / "credentials")
    os.makedirs(creds_dir, exist_ok=True)
    test_creds_path = str(Path(creds_dir) / "google_credentials.json")
    # Write minimal mock credentials for testing purposes
    with open(test_creds_path, "w") as f:
        f.write('{"type": "service_account", "project_id": "test-project", "private_key_id": "test-key", "private_key": "test", "client_email": "test@test-project.iam.gserviceaccount.com", "client_id": "test", "auth_uri": "https://accounts.google.com/o/oauth2/auth", "token_uri": "https://oauth2.googleapis.com/token", "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs", "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/test@test-project.iam.gserviceaccount.com"}')
    os.environ.update({
        "GOOGLE_API_KEY": "test_google_api_key",
        "HUGGINGFACE_TOKEN": "test_hf_token",
        "GOOGLE_APPLICATION_CREDENTIALS": test_creds_path
    })

def teardown_test_credentials():
    """Remove test credentials from environment variables."""
    for key in ["GOOGLE_API_KEY", "HUGGINGFACE_TOKEN", "GOOGLE_APPLICATION_CREDENTIALS"]:
        os.environ.pop(key, None)
