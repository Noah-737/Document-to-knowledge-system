FROM python:3.11-slim

WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -e ".[dev]"

# This image is for build-time checks, no entrypoint needed.
