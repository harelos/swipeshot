# Use the official RunPod PyTorch image as the base
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Upgrade pip and install core system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy our python script into the container
COPY handler.py .

# Trigger a python preload script to download Flux models into the Docker image
# This prevents downloading 20GB of weights every time a serverless container starts
RUN python -c "import torch; from diffusers import FluxPipeline; FluxPipeline.from_pretrained('black-forest-labs/FLUX.1-schnell', torch_dtype=torch.bfloat16)"

# Start the handler when the container runs
CMD [ "python", "-u", "/app/handler.py" ]
