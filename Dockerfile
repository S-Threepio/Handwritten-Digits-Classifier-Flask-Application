# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org Flask opencv-python-headless Pillow requests

# Install CPU-only version of PyTorch and torchvision
RUN pip install torch==2.2.0+cpu torchvision==0.17.0+cpu -f https://download.pytorch.org/whl/torch_stable.html

# Make port 88 available to the world outside this container
EXPOSE 88

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]
