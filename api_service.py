# app/main.py

import logging
import os
import sys
import time
import uuid
from typing import List

# Add current directory to Python path for imports
sys.path.insert(0, "/opt/omniparser")

import cv2  # type: ignore
import numpy as np
import torch  # type: ignore
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

# ─── Configure logging ─────────────────────────────────────────────────────────
os.makedirs("/data/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/data/logs/fastapi_omniparser.log"),
        logging.StreamHandler()  # Also log to console
    ]
)
logger = logging.getLogger("omniparser_api")

# ─── Initialize FastAPI ────────────────────────────────────────────────────────
app = FastAPI(
    title="OmniParser API",
    description="Screen parsing tool for GUI automation",
    version="2.0.0"
)

# Allow CORS from anywhere (adjust in production as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Load OmniParser Models ONCE at startup ─────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Loading models on device: {DEVICE}")

WEIGHTS_PATH = "/data/weights/icon_caption_florence"
DETECT_WEIGHTS = "/data/weights/icon_detect/model.pt"

# Global model variables
detect_model = None
caption_model_processor = None

def get_yolo_model(model_path):
    """Load YOLO model from path"""
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)
        logger.info(f"YOLO model loaded successfully from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load YOLO model: {e}")
        return None

def get_caption_model_processor(model_name="florence2", model_name_or_path=None):
    """Load caption model and processor"""
    try:
        if model_name == "florence2":
            from transformers import AutoProcessor, AutoModelForCausalLM
            processor = AutoProcessor.from_pretrained("microsoft/Florence-2-base", trust_remote_code=True)
            if DEVICE == 'cpu':
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, 
                    torch_dtype=torch.float32, 
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path, 
                    torch_dtype=torch.float16, 
                    trust_remote_code=True
                ).to(DEVICE)
            
            logger.info(f"Florence2 model loaded successfully from {model_name_or_path}")
            return {'model': model, 'processor': processor, 'model_type': 'florence2'}
        else:
            # Fallback to rule-based
            logger.info("Using rule-based captions")
            return {'model_type': 'rule_based'}
    except Exception as e:
        logger.warning(f"Failed to load caption model: {e}, falling back to rule-based")
        return {'model_type': 'rule_based'}

def initialize_models():
    """Initialize models on startup"""
    global detect_model, caption_model_processor
    
    logger.info("Initializing models...")
    
    # Load detection model
    if os.path.exists(DETECT_WEIGHTS):
        logger.info(f"Loading detection model from {DETECT_WEIGHTS}")
        detect_model = get_yolo_model(DETECT_WEIGHTS)
        if detect_model is None:
            logger.error("Failed to load YOLO detection model")
    else:
        logger.error(f"Detection model not found at {DETECT_WEIGHTS}")
    
    # Load caption model
    if os.path.exists(WEIGHTS_PATH):
        logger.info(f"Loading caption model from {WEIGHTS_PATH}")
        caption_model_processor = get_caption_model_processor("florence2", WEIGHTS_PATH)
    else:
        logger.warning(f"Caption model not found at {WEIGHTS_PATH}, using rule-based captions")
        caption_model_processor = {'model_type': 'rule_based'}
    
    logger.info("Model initialization complete")

# ─── Helper: Read raw bytes → PIL Image ────────────────────────────────────────
def read_imagefile_to_pil(data: bytes) -> Image.Image:
    from io import BytesIO
    return Image.open(BytesIO(data)).convert("RGB")

# ─── Pydantic schema for response ─────────────────────────────────────────────
class BoxCaption(BaseModel):
    box: List[float]      # [x1, y1, x2, y2] normalized coordinates
    caption: str
    confidence: float

class ParseResult(BaseModel):
    results: List[BoxCaption]

# ─── POST /api/parse ───────────────────────────────────────────────────────────
@app.post("/api/parse", response_model=ParseResult)
async def parse_image(file: UploadFile = File(...)):
    # Check if models are loaded
    if detect_model is None:
        raise HTTPException(status_code=503, detail="Detection model not loaded")
    
    # 1) Read raw bytes
    try:
        contents = await file.read()
    except Exception as e:
        logger.error(f"Failed to read upload {file.filename}: {e}")
        raise HTTPException(status_code=400, detail="Cannot read uploaded file")

    # 2) Persist raw upload under /data/uploads
    os.makedirs("/data/uploads", exist_ok=True)
    uid = uuid.uuid4().hex[:8]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ext = os.path.splitext(file.filename or "image.png")[1] or ".png"
    save_name = f"upload_{timestamp}_{uid}{ext}"
    save_path = os.path.join("/data/uploads", save_name)
    try:
        with open(save_path, "wb") as f:
            f.write(contents)
        logger.info(f"Saved upload to {save_path}")
    except Exception as e:
        logger.warning(f"Failed to save upload: {e}")

    # 3) Convert bytes → PIL Image
    try:
        img_pil = read_imagefile_to_pil(contents)
        img_np = np.array(img_pil)
    except Exception as e:
        logger.error(f"Invalid image file: {e}")
        raise HTTPException(status_code=400, detail="Invalid image format")

    # 4) Run detection using YOLO
    try:
        # Use the YOLO model to detect objects
        results = detect_model.predict(source=img_pil, conf=0.1, device=DEVICE)
        
        if not results or len(results) == 0:
            logger.info("No objects detected")
            return JSONResponse(content={"results": []})
        
        result = results[0]
        boxes = result.boxes.xyxy.cpu().numpy()  # Get boxes in xyxy format
        confidences = result.boxes.conf.cpu().numpy()  # Get confidence scores
        
        # Normalize coordinates to 0-1 range
        h, w = img_np.shape[:2]
        normalized_boxes = boxes / np.array([w, h, w, h])
        
        # 5) Generate captions for each detected box
        captions = []
        
        if caption_model_processor is None or caption_model_processor.get('model_type') == 'rule_based':
            # Fallback to simple descriptive labels based on box properties
            logger.info("Using rule-based captions based on detection properties")
            for i, (box, conf) in enumerate(zip(normalized_boxes, confidences)):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                area = width * height
                aspect_ratio = width / height if height > 0 else 1
                
                # Generate descriptive labels based on properties
                if area < 0.01:  # Very small
                    size_desc = "small"
                elif area < 0.1:  # Medium
                    size_desc = "medium"
                else:  # Large
                    size_desc = "large"
                
                if aspect_ratio > 2:
                    shape_desc = "wide"
                elif aspect_ratio < 0.5:
                    shape_desc = "tall"
                else:
                    shape_desc = "rectangular"
                
                # Simple position description
                center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
                if center_y < 0.3:
                    pos_desc = "top"
                elif center_y > 0.7:
                    pos_desc = "bottom"
                else:
                    pos_desc = "center"
                
                caption = f"{size_desc} {shape_desc} element in {pos_desc} area"
                captions.append(caption)
        else:
            model = caption_model_processor['model']
            processor = caption_model_processor['processor']
            model_type = caption_model_processor.get('model_type', 'florence2')
            
            for i, box in enumerate(boxes):
                try:
                    # Crop the detected region
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Ensure valid crop coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 <= x1 or y2 <= y1:
                        captions.append(f"invalid_box_{i}")
                        continue
                    
                    cropped_img = img_pil.crop((x1, y1, x2, y2))
                    
                    # Generate caption using Florence2
                    if model_type == 'florence2':
                        prompt = "<CAPTION>"
                        inputs = processor(images=cropped_img, text=prompt, return_tensors="pt").to(DEVICE)
                        with torch.no_grad():
                            generated_ids = model.generate(
                                input_ids=inputs["input_ids"],
                                pixel_values=inputs["pixel_values"],
                                max_new_tokens=20,
                                num_beams=1,
                                do_sample=False
                            )
                        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    else:
                        generated_text = f"object_{i}"
                    
                    captions.append(generated_text.strip() if generated_text.strip() else f"object_{i}")
                    
                except Exception as e:
                    logger.warning(f"Failed to generate caption for box {i}: {e}")
                    captions.append(f"object_{i}")
        
    except Exception as e:
        logger.error(f"Error during inference: {e}")
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # 6) Build response
    out = []
    for box, caption, conf in zip(normalized_boxes, captions, confidences):
        out.append({
            "box": box.tolist(), 
            "caption": caption, 
            "confidence": float(conf)
        })

    logger.info(f"Returning {len(out)} results for {file.filename}")
    return JSONResponse(content={"results": out})

# ─── Simple healthcheck ────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    return {
        "status": "ok", 
        "device": DEVICE,
        "yolo_model_loaded": detect_model is not None,
        "caption_model_loaded": caption_model_processor is not None,
        "weights_path": WEIGHTS_PATH,
        "detect_weights": DETECT_WEIGHTS,
        "detect_model_exists": os.path.exists(DETECT_WEIGHTS),
        "caption_model_exists": os.path.exists(WEIGHTS_PATH)
    }

# ─── Startup event ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Starting OmniParser API...")
    initialize_models()
    logger.info("OmniParser API startup complete")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
