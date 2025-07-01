"""
Image Processing Utilities for PathoDetect+
Simplified version for basic image operations
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
import io

class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.patch_size = config['image_processing']['patch_size']
        self.overlap_ratio = config['image_processing']['overlap_ratio']
    
    def load_image(self, uploaded_file) -> Optional[Image.Image]:
        """Load image from uploaded file"""
        try:
            image = Image.open(uploaded_file)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            print(f"Error loading image: {e}")
            return None
    
    def extract_patches(self, image: Image.Image, patch_size: int = 224) -> List[dict]:
        """Extract patches from the image, returning patch and coordinates"""
        if image is None:
            return []
        
        patches = []
        width, height = image.size
        
        # Calculate stride based on overlap
        stride = int(patch_size * (1 - self.overlap_ratio))
        
        # Extract patches
        for y in range(0, height - patch_size + 1, stride):
            for x in range(0, width - patch_size + 1, stride):
                patch = image.crop((x, y, x + patch_size, y + patch_size))
                patches.append({"patch": patch, "x": x, "y": y})
        
        print(f"Extracted {len(patches)} patches from image")
        return patches
    
    def resize_image(self, image: Image.Image, max_size: int = 1024) -> Image.Image:
        """Resize image while maintaining aspect ratio"""
        width, height = image.size
        
        if width <= max_size and height <= max_size:
            return image
        
        # Calculate new dimensions
        if width > height:
            new_width = max_size
            new_height = int(height * max_size / width)
        else:
            new_height = max_size
            new_width = int(width * max_size / height)
        
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    def normalize_patch(self, patch: Image.Image) -> np.ndarray:
        """Normalize patch for model input"""
        # Convert to numpy array
        patch_array = np.array(patch)
        
        # Normalize to [0, 1]
        patch_array = patch_array.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        patch_array = (patch_array - mean) / std
        
        return patch_array
    
    def create_heatmap(self, image: Image.Image, predictions: List[dict], patch_size: int) -> Image.Image:
        """Create a heatmap overlay on the original image"""
        if not predictions:
            return image
        
        # Create heatmap array
        width, height = image.size
        heatmap = np.zeros((height, width))
        
        # Fill heatmap with predictions
        for pred in predictions:
            x, y = pred['x'], pred['y']
            prob = pred['cancer_prob']
            
            # Ensure coordinates are within bounds
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)
            x = max(0, x)
            y = max(0, y)
            
            heatmap[y:y_end, x:x_end] = prob
        
        # Normalize heatmap
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Convert to PIL Image
        heatmap_img = Image.fromarray((heatmap * 255).astype(np.uint8))
        
        # Resize to match original image
        heatmap_img = heatmap_img.resize((width, height))
        
        return heatmap_img
    
    def overlay_heatmap(self, image: Image.Image, heatmap: Image.Image, alpha: float = 0.7) -> Image.Image:
        """Overlay heatmap on original image"""
        # Convert images to RGBA
        image_rgba = image.convert('RGBA')
        heatmap_rgba = heatmap.convert('RGBA')
        
        # Apply alpha to heatmap
        heatmap_data = np.array(heatmap_rgba)
        heatmap_data[:, :, 3] = (heatmap_data[:, :, 3] * alpha).astype(np.uint8)
        heatmap_rgba = Image.fromarray(heatmap_data)
        
        # Composite images
        result = Image.alpha_composite(image_rgba, heatmap_rgba)
        return result.convert('RGB')
    
    def get_image_info(self, image: Image.Image) -> dict:
        """Get basic information about the image"""
        return {
            'size': image.size,
            'mode': image.mode,
            'format': getattr(image, 'format', 'Unknown'),
            'width': image.width,
            'height': image.height
        } 