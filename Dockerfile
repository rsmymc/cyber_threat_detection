# Use an official Python base image
FROM python:3.12

# Set working directory
WORKDIR /app

# Copy your files
COPY . /app

# Install dependencies
RUN pip install --upgrade pip \
 && pip install -r requirements.txt

# Default command
CMD ["python", "main.py"]
