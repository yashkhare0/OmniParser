FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel
ARG DEBIAN_FRONTEND=noninteractive

ENV CUDA_HOME=/usr/local/cuda \
     TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX" \
     SETUPTOOLS_USE_DISTUTILS=stdlib

RUN conda update conda -y

# Install system dependencies
RUN apt-get -y update && apt-get install -y --no-install-recommends \
         wget \
         build-essential \
         git \
         ninja-build \
         supervisor \
         ca-certificates \
         libsm6 \
         libxext6 \
         libxrender-dev \
         libglib2.0-0 \
         libgl1-mesa-glx \
         libgtk2.0-dev \
         libavcodec-dev \
         libavformat-dev \
         libswscale-dev \
         libjpeg-dev \
         libpng-dev \
         libtiff-dev \
         libdc1394-22-dev \
         curl && \
    rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/program

# Copy OmniParser files into the container
COPY . /opt/program/

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install additional required packages
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    safetensors

# Create necessary directories
RUN mkdir -p /data/logs /data/uploads /data/weights

# Make scripts executable
RUN chmod +x /opt/program/download.sh && \
    chmod +x /opt/program/init.sh

# Copy supervisor configuration
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Verify CUDA setup
RUN python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda if torch.cuda.is_available() else 'N/A')"

# Expose API port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/api/health || exit 1

# Start with supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]

