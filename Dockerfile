# Use an official Python runtime as a parent image
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Copy requirements.txt and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . /app/

# Expose port 8000 for the MCP server
EXPOSE 8000

# Command to run the server
CMD ["python", "main.py"]




