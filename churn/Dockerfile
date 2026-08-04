# Use the official Python image from the Docker Hub
FROM python:3.12-slim

# Install system deps including libgomp for OpenMP support
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the files into the container
COPY . /app

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the notebook to generate the synthetic data and train the models
RUN jupyter nbconvert \
      --to notebook \
      --execute 02-telco-churn-model-training.ipynb \
      --output executed-02-telco-churn-model-training.ipynb

# Expose port 35000 for Flask
EXPOSE 35000

# Command to run the Flask application
CMD ["python", "03-telco-churn-server.py"]
