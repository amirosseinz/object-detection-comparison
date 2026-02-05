# Use the official Python image from Docker Hub
FROM python:3.8-slim

# Set the working directory inside the container
WORKDIR /app

# Install build dependencies and libraries required by numpy
RUN apt-get update && apt-get install -y \
    build-essential \
    libatlas-base-dev \
    gfortran \
    cython3 \
    python3-dev

# Upgrade pip and install Cython
RUN pip install --upgrade pip \
    && pip install cython

# Copy the current directory contents into the container at /app
COPY . /app

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir --prefer-binary -r requirements.txt

# Expose the port that your app will run on (if applicable)
EXPOSE 5000

# Set the default command to run your app (adjust based on your actual app entry point)
CMD ["python", "app.py"]