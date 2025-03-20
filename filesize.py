import pandas as pd

# Load olist_orders_dataset.csv
df = pd.read_csv('data/olist_orders_dataset.csv')

# Define main columns (adjust this list based on your actual dataset)
main_columns = [
    'order_id', 
    'customer_id', 
    'order_status', 
    'order_purchase_timestamp', 
    'order_approved_at', 
    'order_delivered_carrier_date', 
    'order_delivered_customer_date', 
    'order_estimated_delivery_date'
]

# Drop rows where ALL main columns are NaN (completely blank rows)
df = df.dropna(subset=main_columns, how='all')

# Save the cleaned dataset back to olist_orders_dataset.csv
df.to_csv('data/olist_orders_dataset.csv', index=False)

# Note: We'll skip modifying cached_orders.csv here since you'll regenerate it with chatbot.py
print(f"Cleaned olist_orders_dataset.csv: {len(df)} rows remaining")