# Use the Python 3.11 base image
FROM python:3.11

# Install required dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-pip

# Set the PYTHONUNBUFFERED environment variable to True
ENV PYTHONUNBUFFERED True

# Set the working directory inside the container
ENV APP_HOME /app
WORKDIR $APP_HOME

# Copy the application code into the container
COPY . ./

# Install Python dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Adjust Gunicorn settings for improved performance and lower latency
CMD exec gunicorn \
    --bind :$PORT \
    --timeout 30 \
    main:app