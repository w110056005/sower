# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt


# Install any needed packages specified in requirements.txt
RUN pip install --trusted-host pypi.python.org -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app

# Make port 8000 available to the world outside this container
EXPOSE 8000
EXPOSE 8090

# Define environment variable
ENV DJANGO_SETTINGS_MODULE=sower_platform.settings

# Run the command to start Django server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]