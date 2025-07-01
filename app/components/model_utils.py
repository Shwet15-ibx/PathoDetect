"""
Model Utilities for PathoDetect+
Handles model loading, initialization, and management
"""

import streamlit as st
import torch
import os
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from models.resnet_classifier import ResNetClassifier
from models.efficientnet_classifier import EfficientNetClassifier
from models.ensemble_model import EnsembleModel

class ModelManager:
    """Manages model loading and initialization"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = {}
        self.current_model = None
        
    def load_models(self):
        """Load all available models"""
        try:
            print("🔄 Starting model loading...")
            
            # Get weights paths from config (if any)
            resnet_weights = self.config.get('models', {}).get('resnet', {}).get('weights_path', None)
            efficientnet_weights = self.config.get('models', {}).get('efficientnet', {}).get('weights_path', None)

            # Load ResNet model
            print("📦 Loading ResNet-50...")
            resnet_model = ResNetClassifier(
                weights_path=resnet_weights,
                device=str(self.device),
                num_classes=2
            )
            self.models["ResNet-50"] = resnet_model
            print("✅ ResNet-50 loaded successfully")
            
            # Load EfficientNet model
            print("📦 Loading EfficientNet-B4...")
            efficientnet_model = EfficientNetClassifier(
                weights_path=efficientnet_weights,
                device=str(self.device),
                num_classes=2
            )
            self.models["EfficientNet-B4"] = efficientnet_model
            print("✅ EfficientNet-B4 loaded successfully")
            
            # Create ensemble model
            print("📦 Creating ensemble model...")
            ensemble_model = EnsembleModel(
                resnet_model=resnet_model,
                efficientnet_model=efficientnet_model,
                weights=(0.6, 0.4)
            )
            self.models["Ensemble"] = ensemble_model
            print("✅ Ensemble model created successfully")
            
            # Set default model
            self.current_model = "ResNet-50"
            print(f"🎯 Default model set to: {self.current_model}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def get_model(self, model_name=None):
        """Get a specific model or current model"""
        if model_name is None:
            model_name = self.current_model
        
        if model_name in self.models:
            return self.models[model_name]
        else:
            return None
    
    def get_available_models(self):
        """Get list of available models"""
        return list(self.models.keys())
    
    def switch_model(self, model_name):
        """Switch to a different model"""
        if model_name in self.models:
            self.current_model = model_name
            return True
        return False
    
    def predict_patches(self, patches, confidence_threshold=0.5):
        """Predict cancer probability for patches, including coordinates if available"""
        if not self.models:
            return []
        
        model = self.get_model()
        if model is None:
            return []
        
        predictions = []
        for i, patch_info in enumerate(patches):
            try:
                # Support both old (Image) and new (dict) patch formats
                if isinstance(patch_info, dict):
                    patch = patch_info.get('patch', patch_info)
                    x = patch_info.get('x', None)
                    y = patch_info.get('y', None)
                else:
                    patch = patch_info
                    x = None
                    y = None
                # Convert patch to PIL Image if needed
                if hasattr(patch, 'convert'):
                    pil_image = patch
                else:
                    from PIL import Image
                    import numpy as np
                    if isinstance(patch, np.ndarray):
                        pil_image = Image.fromarray(patch)
                    else:
                        pil_image = patch
                # Get prediction
                confidence = model.predict(pil_image)
                pred_dict = {
                    'patch_id': i,
                    'cancer_prob': confidence,
                    'prediction': 'malignant' if confidence >= confidence_threshold else 'benign',
                    'model_used': self.current_model
                }
                if x is not None and y is not None:
                    pred_dict['x'] = x
                    pred_dict['y'] = y
                predictions.append(pred_dict)
            except Exception as e:
                st.error(f"Error predicting patch {i}: {e}")
                predictions.append({
                    'patch_id': i,
                    'cancer_prob': 0.0,
                    'prediction': 'error',
                    'model_used': self.current_model
                })
        return predictions

def initialize_models(config):
    """Initialize and load models"""
    print("🚀 Initializing models...")
    try:
        print("📦 Creating new ModelManager...")
        model_manager = ModelManager(config)
        print("📦 ModelManager created, loading models...")
        if model_manager.load_models():
            print("✅ Models loaded successfully!")
            return model_manager
        else:
            print("❌ Failed to load models")
            return None
    except Exception as e:
        print(f"❌ Error initializing models: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_model_status():
    """Get current model status"""
    if 'model_manager' in st.session_state and st.session_state.model_manager is not None:
        # Check if models are actually loaded
        if st.session_state.model_manager.models and st.session_state.model_manager.current_model:
            return True, st.session_state.model_manager.current_model
        else:
            # Models exist but not loaded, try to load them
            print("🔄 Models exist but not loaded, attempting to load...")
            if st.session_state.model_manager.load_models():
                return True, st.session_state.model_manager.current_model
            else:
                return False, None
    return False, None 