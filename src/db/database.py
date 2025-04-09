# src/db/database.py
import os
import logging
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
# Import the CORRECT variable name and other needed config
from src.core.config import SQLALCHEMY_DATABASE_URL, DATA_DIR # Removed SQLALCHEMY_ECHO if not used here

logger = logging.getLogger(__name__)

# Ensure the directory for the database exists if it's SQLite
# This check might be redundant if config.py already does it, but harmless
if SQLALCHEMY_DATABASE_URL.startswith("sqlite:///"):
    db_path = SQLALCHEMY_DATABASE_URL.split("///")[1]
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        logger.info(f"Creating database directory: {db_dir}")
        os.makedirs(db_dir, exist_ok=True)
    # Also ensure the base DATA_DIR exists (if db is inside it)
    if not os.path.exists(DATA_DIR):
         os.makedirs(DATA_DIR, exist_ok=True)


try:
    logger.info(f"Attempting to create engine for database: {SQLALCHEMY_DATABASE_URL}")
    # Use the correct variable name here
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False} if SQLALCHEMY_DATABASE_URL.startswith("sqlite") else {}
        # Removed echo=SQLALCHEMY_ECHO unless you define and import SQLALCHEMY_ECHO in config.py
    )
    logger.info(f"SQLAlchemy engine created successfully for: {SQLALCHEMY_DATABASE_URL.split('///')[-1]}")
except Exception as e:
    logger.error(f"Failed to create SQLAlchemy engine: {e}", exc_info=True)
    raise

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
logger.info("SQLAlchemy SessionLocal created.")

# Use the updated import location
Base = declarative_base()
logger.info("SQLAlchemy declarative base created.")

def get_db():
    """Dependency to get a database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize the database tables."""
    logger.info("Initializing database tables...")
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize database tables: {e}", exc_info=True)