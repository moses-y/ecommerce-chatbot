# Dockerfile

# Use an official Python runtime matching your development environment
FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set the working directory
WORKDIR /app

# Install uv first
# Copy only requirements to leverage Docker cache
COPY requirements.txt .
RUN pip install uv && \
    uv pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes app.py, src/, tests/, etc.
COPY . .

# --- Additions based on your code ---
# Create the data directory inside the container
RUN mkdir -p /app/data
# Copy the necessary policy file into the container's data directory
COPY data/policies.json /app/data/policies.json

# Copy the assets directory into the container
COPY assets /app/assets
# --- End Additions ---

# Expose the port Gradio runs on (defined in gradio_app.py launch())
EXPOSE 7860

# Define the command to run your application using the root app.py script
CMD ["python", "app.py"]