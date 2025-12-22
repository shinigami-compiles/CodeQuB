# Use Python 3.10 explicitly (TensorFlow compatible)
FROM python:3.10.13-slim

# Environment safety
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Working directory inside container
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (cache optimization)
COPY requirements.txt .

# Upgrade pip
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Render uses port 10000
EXPOSE 10000

# Start Flask app via Gunicorn
CMD [
  "gunicorn",
  "app:app",
  "--bind", "0.0.0.0:10000",
  "--timeout", "180",
  "--workers", "1"
]

