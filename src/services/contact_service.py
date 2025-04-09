# src/services/contact_service.py
import logging
from sqlalchemy.orm import Session
from src.db.database import SessionLocal
from src.db.models import ContactRequest
import datetime

logger = logging.getLogger(__name__)

def save_contact_request(full_name: str, email: str, phone_number: str = None, notes: str = None) -> bool:
    """
    Saves a customer's contact request to the database.

    Args:
        full_name: Customer's full name.
        email: Customer's email address.
        phone_number: Customer's phone number (optional).
        notes: Any additional notes about the request (optional).

    Returns:
        True if the request was saved successfully, False otherwise.
    """
    logger.info(f"Attempting to save contact request for email: {email}")
    db: Session = SessionLocal()
    try:
        # Basic validation (can be enhanced in agent/utils)
        if not full_name or not email:
            logger.warning("Attempted to save contact request with missing name or email.")
            return False

        new_request = ContactRequest(
            full_name=full_name,
            email=email,
            phone_number=phone_number,
            request_timestamp=datetime.datetime.now(datetime.UTC), # Ensure UTC timestamp
            notes=notes
        )
        db.add(new_request)
        db.commit()
        db.refresh(new_request) # To get the generated ID if needed
        logger.info(f"Contact request saved successfully with ID: {new_request.id}")
        return True
    except Exception as e:
        logger.error(f"Database error saving contact request for {email}: {e}", exc_info=True)
        db.rollback() # Rollback the transaction on error
        return False
    finally:
        db.close()
        logger.debug(f"Database session closed for contact request save: {email}")