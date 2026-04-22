# Use a professional Python base image
FROM python:3.12-slim

# Set environment variables for non-interactive installs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# 🧠 Reassemble the expert model on build (if needed)
RUN python3 -c "from server.app import _reassemble_model; _reassemble_model()"

# Expose the API port
EXPOSE 7860

# Command to run the application
CMD ["python3", "server/app.py"]