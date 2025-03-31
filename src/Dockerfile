FROM --platform=linux/amd64 python:3.9-slim as build
# Set the working directory in the container

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0

RUN mkdir -p /app

WORKDIR /app

# Copy all server code
COPY . /app

# Install the required packages
RUN pip install -r requirements.txt
EXPOSE 8000

CMD ["python", "index.py"]
