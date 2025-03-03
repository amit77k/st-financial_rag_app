# Build stage
FROM python:3.9-slim as builder

WORKDIR /st-financial_rag_app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /app
