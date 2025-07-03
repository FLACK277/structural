# Use Python 3.9 slim image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies and clean up in one layer
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install only essential packages - remove heavy ones
RUN python -m pip install --upgrade pip setuptools wheel --no-cache-dir
RUN pip install --no-cache-dir -r requirements.txt \
    && pip cache purge \
    && rm -rf ~/.cache/pip

# Copy the rest of the application
COPY . .

# Expose port (adjust if your app uses a different port)
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
