# Use Python 3.10 as the base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install uv for faster dependency installation
RUN pip install uv

# Copy requirements file
COPY requirements3.txt .

# Install dependencies using uv (with --system flag to install in non-virtual environment)
RUN uv pip install --system -r requirements3.txt

# Copy the application code
COPY . /app

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=main.py

# Expose the port the app runs on
EXPOSE 5001

# Command to run the application
CMD ["python", "src/api/main.py"]
