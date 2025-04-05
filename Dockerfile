# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set environment variable to disable output buffering
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project code
COPY . .

# Expose the port that the FastAPI app will run on
EXPOSE 8080

# Command to run the FastAPI API using Uvicorn
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8080"]
