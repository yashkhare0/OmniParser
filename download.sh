#!/bin/bash

set -e

echo "Downloading OmniParser V2 models..."

# Define folder structure and create folders
mkdir -p weights/icon_detect
mkdir -p weights/icon_caption_florence
mkdir -p weights/icon_caption_blip2

# Download V2 models from HuggingFace
echo "Downloading icon detection model..."
wget -O weights/icon_detect/train_args.yaml "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/train_args.yaml"
wget -O weights/icon_detect/model.pt "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.pt"
wget -O weights/icon_detect/model.yaml "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_detect/model.yaml"

echo "Downloading Florence caption model..."
wget -O weights/icon_caption_florence/config.json "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/config.json"
wget -O weights/icon_caption_florence/generation_config.json "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/generation_config.json"
wget -O weights/icon_caption_florence/model.safetensors "https://huggingface.co/microsoft/OmniParser-v2.0/resolve/main/icon_caption/model.safetensors"

echo "All required model files downloaded successfully."

# Verify files exist
if [ -f "weights/icon_detect/model.pt" ] && [ -f "weights/icon_caption_florence/model.safetensors" ]; then
    echo "✓ All required model files are present"
else
    echo "Error: Some required files are missing"
    exit 1
fi

echo "Model download complete!"