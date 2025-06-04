# OmniParser Model Management Guide

This guide explains how to manage OmniParser models for experimentation, fine-tuning, and custom model development.

## 📁 Directory Structure

```
OmniParser/
├── data/
│   ├── weights/           # Model files (1.1GB)
│   │   ├── icon_detect/   # YOLO object detection models
│   │   │   ├── model.pt
│   │   │   ├── model.yaml
│   │   │   └── train_args.yaml
│   │   ├── icon_caption_florence/  # Florence caption models
│   │   │   ├── config.json
│   │   │   ├── generation_config.json
│   │   │   └── model.safetensors
│   │   └── icon_caption_blip2/     # (Ready for BLIP2 models)
│   ├── logs/              # API and initialization logs
│   └── uploads/           # Temporary image uploads
├── Dockerfile
├── api_service.py
├── download.sh
├── init.sh
└── MODEL_MANAGEMENT.md    # This file
```

## 🚀 Quick Start

### 1. **Initial Setup** (Already Done)
The models are automatically downloaded and mounted locally:
```bash
make dev  # Models download to ./OmniParser/data/weights/
```

### 2. **Verify Models**
Check that models are properly mounted:
```bash
# Check model files
ls -la OmniParser/data/weights/
du -sh OmniParser/data/weights/  # Should show ~1.1GB
```

## 🔧 Model Management

### **Adding New Models**
To add custom or new models:

1. **Place model files** in appropriate directories:
   ```bash
   # For detection models
   cp your_custom_model.pt OmniParser/data/weights/icon_detect/
   
   # For caption models
   cp your_florence_model.safetensors OmniParser/data/weights/icon_caption_florence/
   ```

2. **Update configuration** if needed:
   - Edit `icon_detect/model.yaml` for detection model config
   - Edit `icon_caption_florence/config.json` for caption model config

3. **Restart container** to load new models:
   ```bash
   make stop
   make dev
   ```

### **Model Experimentation**
Since models are locally mounted, you can:

- **Swap models** by replacing files in `data/weights/`
- **A/B test** by renaming model files
- **Version control** by using subdirectories or naming conventions
- **Backup models** by copying the entire `data/weights/` directory

### **Fine-tuning Workflow**
1. **Extract current models** for fine-tuning:
   ```bash
   cp OmniParser/data/weights/icon_detect/model.pt ./my_finetuning/base_model.pt
   ```

2. **Fine-tune** using your preferred training setup

3. **Replace models** with fine-tuned versions:
   ```bash
   cp ./my_finetuning/finetuned_model.pt OmniParser/data/weights/icon_detect/model.pt
   ```

4. **Test** the new model:
   ```bash
   make test-omniparser
   ```

## 📊 Supported Model Formats

### **Detection Models** (`icon_detect/`)
- **Format**: PyTorch (`.pt`)
- **Type**: YOLO-based object detection
- **Size**: ~40MB
- **Config**: `model.yaml`, `train_args.yaml`

### **Caption Models** (`icon_caption_florence/`)
- **Format**: Safetensors (`.safetensors`)
- **Type**: Florence-2 vision-language model
- **Size**: ~1.08GB
- **Config**: `config.json`, `generation_config.json`

### **Future Models** (`icon_caption_blip2/`)
- **Format**: Safetensors/PyTorch
- **Type**: BLIP2 models (ready for future use)
- **Config**: Standard HuggingFace format

## 🔄 Model Download & Caching

### **How Downloads Work**
1. **Check**: `init.sh` checks if models exist in `/data/weights/`
2. **Download**: If missing, downloads from HuggingFace using `download.sh`
3. **Cache**: Models persist in local `./OmniParser/data/weights/`
4. **Reuse**: Subsequent starts skip download (warm start)

### **Force Re-download**
To force fresh model downloads:
```bash
# Remove local models
rm -rf OmniParser/data/weights/*

# Restart container (will trigger download)
make stop
make dev
```

### **Offline Mode**
Once models are downloaded, OmniParser works offline:
- Models cached locally in `./OmniParser/data/weights/`
- No internet required for subsequent starts
- Perfect for air-gapped environments

## 🧪 Testing & Validation

### **Test Model Loading**
```bash
# Test health endpoint (verifies models loaded)
make test-omniparser-health

# Test full pipeline
make test-omniparser
```

### **Check Model Status**
```bash
# View model loading logs
docker-compose logs omniparser | grep -i model

# Check API status with model info
curl http://localhost:8081/api/health
```

## 🚨 Troubleshooting

### **Models Not Loading**
1. **Check files exist**:
   ```bash
   ls -la OmniParser/data/weights/icon_detect/model.pt
   ls -la OmniParser/data/weights/icon_caption_florence/model.safetensors
   ```

2. **Check permissions**:
   ```bash
   chmod -R 755 OmniParser/data/
   ```

3. **Check logs**:
   ```bash
   docker-compose logs omniparser
   cat OmniParser/data/logs/omniparser-api.log
   ```

### **Disk Space Issues**
Models require ~1.1GB of space:
```bash
# Check available space
df -h .

# Clean up old Docker volumes if needed
docker volume prune
```

## 💡 Best Practices

1. **Backup**: Keep a backup of working models before experimenting
2. **Versioning**: Use clear naming for custom models (`model_v1.pt`, `finetuned_20241203.pt`)
3. **Documentation**: Document model changes in comments or README files
4. **Testing**: Always test after model changes using `make test-omniparser`
5. **Monitoring**: Check logs for model loading issues during development

## 🔗 Resources

- **OmniParser Repository**: https://github.com/microsoft/OmniParser
- **HuggingFace Models**: https://huggingface.co/microsoft/OmniParser-v2.0
- **Florence-2 Documentation**: https://huggingface.co/microsoft/Florence-2-large
- **YOLO Documentation**: https://docs.ultralytics.com/ 