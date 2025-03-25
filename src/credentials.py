# utils/credentials.py
import os
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

REQUIRED_CREDENTIALS = {
    "GOOGLE_API_KEY": "Google API key",
    "GOOGLE_APPLICATION_CREDENTIALS": "Google Application Credentials path",
    "HUGGINGFACE_TOKEN": "Hugging Face API token"
}

def verify_credentials(required_creds: List[str] = None) -> Dict[str, bool]:
    """
    Verify that required credentials are properly configured.
    Args:
        required_creds: List of credential keys to verify. If None, verify all.
    Returns:
        Dictionary of credential verification results
    """
    if required_creds is None:
        required_creds = REQUIRED_CREDENTIALS.keys()
    
    results = {}
    
    for cred_key in required_creds:
        cred_value = os.getenv(cred_key)
        if not cred_value:
            logger.warning(f"{REQUIRED_CREDENTIALS[cred_key]} not found in environment variables")
            results[cred_key] = False
            continue
            
        # Special handling for Google Application Credentials file path
        if cred_key == "GOOGLE_APPLICATION_CREDENTIALS":
            if not os.path.exists(cred_value):
                logger.error(f"Credentials file not found at: {cred_value}")
                results[cred_key] = False
                continue
                
        results[cred_key] = True
        logger.info(f"{REQUIRED_CREDENTIALS[cred_key]} verified successfully")
    
    return results