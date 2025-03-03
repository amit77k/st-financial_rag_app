# Build stage
FROM python:3.9-slim as builder

WORKDIR /st-financial_rag_app

# Copy requirements and install dependencies
COPY requirements.txt .
source /home/adminuser/venv/bin/activate  # for Linux/macOS
venv\Scripts\activate  # for Windows

pip install pdfplumber

pip install pdfplumber
pip show pdfplumber
RUN pip install --no-cache-dir -r requirements.txt

# Runtime stage
FROM python:3.9-slim

WORKDIR /st-financial_rag_app
