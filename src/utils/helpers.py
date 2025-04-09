# src/utils/helpers.py
import re
import logging
import string

logger = logging.getLogger(__name__)

# Update the pattern and compile with debug flag
ORDER_ID_PATTERN_SEARCH = re.compile(r'[a-zA-Z0-9]{32}')  # Removed capture group parentheses

def extract_order_id(text: str) -> str | None:
    """
    Extracts the first 32-character alphanumeric string (likely order ID) from text.
    Prioritizes checking if the entire string is exactly 32 chars without regex first.
    """
    if not isinstance(text, str) or not text:
        logger.debug("Input text is not a non-empty string.")
        return None

    logger.debug("--- Inside extract_order_id ---")
    logger.debug(f"Input text: '{text}'")
    logger.debug(f"Input type: {type(text)}")

    try:
        # 1. Manual check first - for exact matches
        text = text.strip()
        text_len = len(text)
        valid_chars = set(string.ascii_letters + string.digits)
        
        # Only do exact match if the string is exactly 32 chars
        if text_len == 32 and all(c in valid_chars for c in text):
            logger.debug(f"Extracted order ID (exact match): {text}")
            return text
            
        # 2. If not exact match, search within the string
        logger.debug("Trying regex search within string")
        # Use findall instead of search to get all matches
        matches = ORDER_ID_PATTERN_SEARCH.findall(text)
        logger.debug(f"Found matches: {matches}")
        
        if matches:
            # Take the first match
            order_id = matches[0]
            logger.debug(f"Extracted order ID (regex match): {order_id}")
            return order_id

        logger.debug("No valid order ID found")
        return None

    except Exception as e:
        logger.error(f"Exception during operations in extract_order_id: {e}", exc_info=True)
        return None
    finally:
        logger.debug("--- Exiting extract_order_id ---")