# src/db/models.py
import datetime
from sqlalchemy import Column, String, DateTime, Integer, Text
from sqlalchemy.orm import declarative_base
from src.db.database import Base
from src.core.config import ORDERS_TABLE_NAME, CONTACTS_TABLE_NAME

class Order(Base):
    """SQLAlchemy model for orders."""
    __tablename__ = ORDERS_TABLE_NAME

    # Assuming order_id is the unique identifier
    order_id = Column(String(32), primary_key=True, index=True)
    customer_id = Column(String(32), index=True) # Index for potential lookups
    order_status = Column(String(50))

    # Use DateTime for timestamp fields
    # Ensure the CSV loading handles date parsing correctly
    order_purchase_timestamp = Column(DateTime, nullable=True)
    order_approved_at = Column(DateTime, nullable=True)
    order_delivered_carrier_date = Column(DateTime, nullable=True)
    order_delivered_customer_date = Column(DateTime, nullable=True)
    order_estimated_delivery_date = Column(DateTime, nullable=True)

    # Add any other relevant columns from your CSV if needed
    # Example: payment_type = Column(String(50))

    def __repr__(self):
        return f"<Order(order_id='{self.order_id}', status='{self.order_status}')>"

class ContactRequest(Base):
    """SQLAlchemy model for storing human representative contact requests."""
    __tablename__ = CONTACTS_TABLE_NAME

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    full_name = Column(String(255), nullable=False)
    email = Column(String(255), nullable=False, index=True) # Index for potential lookups
    phone_number = Column(String(50), nullable=True) # Allow phone to be optional maybe?
    request_timestamp = Column(DateTime, default=datetime.datetime.now(datetime.UTC)) # Store timestamp
    notes = Column(Text, nullable=True) # Optional field for context

    def __repr__(self):
        return f"<ContactRequest(id={self.id}, email='{self.email}', timestamp='{self.request_timestamp}')>"