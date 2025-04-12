# app.py is the main entry point for the Gradio application.
import logging
import os
from pathlib import Path
from sqlalchemy import text, select, func
from src.ui.gradio_app import main
from src.db.setup_db import create_tables, load_orders_from_csv
from src.db.database import SessionLocal
from src.db.models import Order  # Import the Order model

def setup_logging():
    """Configure logging for the application"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def ensure_database():
    """Ensure database is set up properly"""
    logger = logging.getLogger(__name__)

    try:
        create_tables()
        db = SessionLocal()
        try:
            # Use SQLAlchemy's ORM query instead of raw SQL
            order_count = db.query(func.count(Order.order_id)).scalar()

            if order_count == 0:
                logger.info("No orders found in database, loading initial data...")
                load_orders_from_csv(db)
            else:
                logger.info(f"Database already contains {order_count} orders")
        finally:
            db.close()
    except Exception as e:
        logger.error(f"Database setup error: {e}", exc_info=True)
        raise

def verify_database():
    """Verify database setup and permissions"""
    logger = logging.getLogger(__name__)

    try:
        # Check data directory permissions
        data_dir = "/app/data" if os.environ.get('HF_SPACE') == 'true' else "data"
        os.makedirs(data_dir, exist_ok=True)

        # Verify write permissions
        test_file = os.path.join(data_dir, "test_write.tmp")
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            logger.info("Data directory is writable")
        except Exception as e:
            logger.error(f"Data directory write test failed: {e}")
            raise

        # Create and verify database
        create_tables()
        db = SessionLocal()
        try:
            # Verify orders table
            order_count = db.query(func.count(Order.order_id)).scalar()
            logger.info(f"Orders table verified with {order_count} records")

            # Verify contact_requests table
            from src.db.models import ContactRequest
            contact_count = db.query(func.count(ContactRequest.id)).scalar()
            logger.info(f"ContactRequest table verified with {contact_count} records")

            if order_count == 0:
                logger.info("Loading initial order data...")
                load_orders_from_csv(db)
        finally:
            db.close()

    except Exception as e:
        logger.error(f"Database verification failed: {e}")
        raise

if __name__ == "__main__":
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        os.makedirs("assets", exist_ok=True)

        # Always ensure database is set up
        ensure_database()
        verify_database()

        # Launch the Gradio app
        logger.info("Starting Gradio application...")
        main()
    except Exception as e:
        logger.error(f"Application startup error: {e}", exc_info=True)
        raise