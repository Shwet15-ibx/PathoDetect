"""
TCGA-BRCA Dataset Preprocessing Script
- Organizes WSIs and extracts patches for model training/inference
- Place WSIs in data/raw/ and output patches to data/processed/
"""
import os
from pathlib import Path
import openslide
import numpy as np
from PIL import Image

def extract_patches_from_wsi(wsi_path, output_dir, patch_size=224, overlap=0.5, tissue_threshold=0.7):
    """
    Extracts patches from a whole-slide image (WSI).
    Args:
        wsi_path (str): Path to the WSI file.
        output_dir (str): Directory to save patches.
        patch_size (int): Size of each patch (square).
        overlap (float): Overlap ratio between patches.
        tissue_threshold (float): Minimum tissue fraction to keep a patch.
    """
    slide = openslide.OpenSlide(wsi_path)
    width, height = slide.dimensions
    stride = int(patch_size * (1 - overlap))
    os.makedirs(output_dir, exist_ok=True)
    patch_id = 0
    for y in range(0, height - patch_size + 1, stride):
        for x in range(0, width - patch_size + 1, stride):
            patch = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            patch_np = np.array(patch)
            # Simple tissue filter: keep if mean pixel < 220 (not background)
            if np.mean(patch_np) < 220:
                patch.save(os.path.join(output_dir, f"patch_{patch_id:06d}.png"))
                patch_id += 1
    print(f"Extracted {patch_id} patches from {os.path.basename(wsi_path)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="TCGA-BRCA WSI Patch Extraction")
    parser.add_argument('--input_dir', type=str, default="../../data/raw", help='Directory with WSIs')
    parser.add_argument('--output_dir', type=str, default="../../data/processed", help='Directory for patches')
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--overlap', type=float, default=0.5)
    args = parser.parse_args()

    wsi_files = list(Path(args.input_dir).glob("*.svs")) + list(Path(args.input_dir).glob("*.tif"))
    for wsi_path in wsi_files:
        extract_patches_from_wsi(str(wsi_path), args.output_dir, args.patch_size, args.overlap) 