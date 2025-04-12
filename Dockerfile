# Dockerfile
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PYTHONPATH=/app
ENV HF_SPACE=true

# Set the working directory
WORKDIR /app

# Install requirements
COPY requirements.txt .
RUN pip install uv && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories and ensure proper permissions
RUN mkdir -p /app/data /app/assets && \
    chmod -R 777 /app/data

# Copy data files with explicit permissions
COPY data/policies.json /app/data/policies.json
COPY data/cached_orders.csv /app/data/cached_orders.csv
COPY assets /app/assets

# Set permissions for data files
RUN chmod 644 /app/data/policies.json /app/data/cached_orders.csv

# Expose Gradio port
EXPOSE 7860

# Initialize database and start app
CMD ["sh", "-c", "python -m src.db.setup_db && python app.py"]