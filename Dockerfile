# Use the official Python image as a base image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY main.py .
COPY data_processing.py .
COPY model_training.py .
COPY model_classes.py .
COPY model_evaluation.py .
COPY model_prediction.py .
COPY utils.py .
COPY config.json .

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Specify the default command to run when the container starts
CMD ["python", "main.py"]
