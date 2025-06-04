#!/bin/bash

set -e

echo "OmniParser Initialization Script"
echo "================================="

# Create necessary directories
mkdir -p /data/weights/icon_detect
mkdir -p /data/weights/icon_caption_florence
mkdir -p /data/weights/icon_caption_blip2
mkdir -p /data/logs
mkdir -p /data/uploads

# Check if models are already present (mounted from volume)
if [ -f "/data/weights/icon_detect/model.pt" ] && [ -f "/data/weights/icon_caption_florence/model.safetensors" ]; then
    echo "✓ Models found in mounted volume, skipping download"
else
    echo "Downloading models..."
    
    # Download models using the download script
    cd /opt/omniparser
    ./download.sh
    
    # Copy downloaded models to data directory
    if [ -d "weights" ]; then
        echo "Copying models to /data/weights..."
        cp -r weights/* /data/weights/
    fi
    
    echo "✓ Models copied successfully"
fi

echo "✓ Model initialization complete"

# Verify critical models exist
if [ ! -f "/data/weights/icon_detect/model.pt" ]; then
    echo "⚠️  Warning: Detection model not found, API will run in limited mode"
fi

if [ ! -f "/data/weights/icon_caption_florence/model.safetensors" ]; then
    echo "⚠️  Warning: Caption model not found, will use rule-based captions"
fi

# Start the API service
echo "Starting OmniParser API service..."
exec "$@" 