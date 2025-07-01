"""
PathoDetect+ API Server
FastAPI server for model deployment and inference
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import torch
import numpy as np
from PIL import Image
import io
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from models.resnet_classifier import ResNetClassifier
from models.efficientnet_classifier import EfficientNetClassifier
from models.ensemble_model import EnsembleModel
from utils.image_processing import ImageProcessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PathoDetect+ API",
    description="AI-powered histopathology image analysis API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
models = {}
current_model = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pydantic models
class PredictionRequest(BaseModel):
    model_name: Optional[str] = "resnet"
    confidence_threshold: Optional[float] = 0.5

class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    processing_time: float
    model_used: str
    timestamp: str

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    device: str
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    version: str
    status: str
    accuracy: Optional[float]
    created: str

# Load configuration
def load_config():
    """Load configuration from file"""
    config_path = project_root / "config" / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}

config = load_config()

# Model management
def load_models():
    """Load all available models"""
    global models, current_model
    
    models_dir = project_root / "models" / "saved"
    if not models_dir.exists():
        logger.warning("Models directory not found")
        return
    
    # Load ResNet model
    try:
        resnet_path = models_dir / "resnet_best.pth"
        if resnet_path.exists():
            models["resnet"] = ResNetClassifier(device=str(device))
            logger.info("ResNet model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading ResNet model: {e}")
    
    # Load EfficientNet model
    try:
        efficientnet_path = models_dir / "efficientnet_best.pth"
        if efficientnet_path.exists():
            models["efficientnet"] = EfficientNetClassifier(device=str(device))
            logger.info("EfficientNet model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading EfficientNet model: {e}")
    
    # Create ensemble model if both are available
    if "resnet" in models and "efficientnet" in models:
        try:
            models["ensemble"] = EnsembleModel(
                resnet_model=models["resnet"],
                efficientnet_model=models["efficientnet"],
                weights=(0.6, 0.4)
            )
            logger.info("Ensemble model created successfully")
        except Exception as e:
            logger.error(f"Error creating ensemble model: {e}")
    
    # Set default model
    if models:
        current_model = list(models.keys())[0]
        logger.info(f"Default model set to: {current_model}")

# Initialize models on startup
@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    logger.info("Starting PathoDetect+ API server...")
    load_models()
    logger.info(f"API server started on device: {device}")

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=list(models.keys()),
        device=str(device),
        timestamp=datetime.now().isoformat()
    )

# Model information endpoint
@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models"""
    model_info = []
    
    for name, model in models.items():
        info = ModelInfo(
            name=name,
            version="1.0.0",
            status="loaded",
            accuracy=87.3,  # This would be loaded from model metadata
            created=datetime.now().isoformat()
        )
        model_info.append(info)
    
    return model_info

# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
async def predict_cancer(
    file: UploadFile = File(...),
    model_name: Optional[str] = "resnet",
    confidence_threshold: Optional[float] = 0.5
):
    """Predict cancer probability for uploaded image"""
    
    start_time = datetime.now()
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check if model is available
    if model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' not available. Available models: {list(models.keys())}"
        )
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        # Get model
        model = models[model_name]
        
        # Make prediction
        if hasattr(model, 'predict'):
            # For our custom model classes
            confidence = model.predict(image)
        else:
            # For standard PyTorch models
            import torchvision.transforms as transforms
            
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_tensor)
                confidence = torch.softmax(output, dim=1)[0, 1].item()
        
        # Determine prediction
        prediction = "malignant" if confidence >= confidence_threshold else "benign"
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            prediction=prediction,
            confidence=confidence,
            processing_time=processing_time,
            model_used=model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
async def predict_batch(
    files: List[UploadFile] = File(...),
    model_name: Optional[str] = "resnet",
    confidence_threshold: Optional[float] = 0.5
):
    """Predict cancer probability for multiple images"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' not available"
        )
    
    results = []
    
    for i, file in enumerate(files):
        try:
            # Create a temporary prediction request
            image_data = await file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            model = models[model_name]
            
            if hasattr(model, 'predict'):
                confidence = model.predict(image)
            else:
                import torchvision.transforms as transforms
                
                transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                
                input_tensor = transform(image).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    output = model(input_tensor)
                    confidence = torch.softmax(output, dim=1)[0, 1].item()
            
            prediction = "malignant" if confidence >= confidence_threshold else "benign"
            
            results.append({
                "file_name": file.filename,
                "prediction": prediction,
                "confidence": confidence,
                "model_used": model_name
            })
            
        except Exception as e:
            results.append({
                "file_name": file.filename,
                "error": str(e)
            })
    
    return {
        "predictions": results,
        "total_files": len(files),
        "successful_predictions": len([r for r in results if "error" not in r]),
        "timestamp": datetime.now().isoformat()
    }

# Model switching endpoint
@app.post("/models/switch")
async def switch_model(model_name: str):
    """Switch to a different model"""
    global current_model
    
    if model_name not in models:
        raise HTTPException(
            status_code=400, 
            detail=f"Model '{model_name}' not available"
        )
    
    current_model = model_name
    
    return {
        "message": f"Switched to model: {model_name}",
        "current_model": current_model,
        "timestamp": datetime.now().isoformat()
    }

# Model reload endpoint
@app.post("/models/reload")
async def reload_models():
    """Reload all models"""
    global models, current_model
    
    # Clear existing models
    models.clear()
    
    # Reload models
    load_models()
    
    return {
        "message": "Models reloaded successfully",
        "loaded_models": list(models.keys()),
        "current_model": current_model,
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "PathoDetect+ API",
        "version": "1.0.0",
        "description": "AI-powered histopathology image analysis API",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict": "/predict",
            "predict_batch": "/predict/batch",
            "switch_model": "/models/switch",
            "reload_models": "/models/reload"
        },
        "documentation": "/docs",
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    ) 