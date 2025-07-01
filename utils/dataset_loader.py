"""
BreaKHis Dataset Loader for PathoDetect+
Handles loading and preprocessing of BreaKHis histopathology dataset
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class BreaKHisDataset(Dataset):
    """Dataset class for BreaKHis histopathology images"""
    
    def __init__(self, 
                 data_root: str,
                 folds_csv: str,
                 fold: int = 1,
                 split: str = 'train',
                 magnification: str = '100',
                 transform: Optional[transforms.Compose] = None,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize BreaKHis dataset
        
        Args:
            data_root: Root directory containing BreaKHis_v1 folder
            folds_csv: Path to Folds.csv file
            fold: Fold number (1-5)
            split: 'train' or 'test'
            magnification: '40', '100', '200', or '400'
            transform: Optional torchvision transforms
            target_size: Target image size (width, height)
        """
        self.data_root = Path(data_root)
        self.fold = fold
        self.split = split
        self.magnification = magnification
        self.target_size = target_size
        
        # Load folds data
        self.folds_df = pd.read_csv(folds_csv)
        
        # Filter data based on fold, split, and magnification
        self.data = self.folds_df[
            (self.folds_df['fold'] == fold) & 
            (self.folds_df['grp'] == split) & 
            (self.folds_df['mag'] == int(magnification))
        ].copy()
        
        # Create labels (0: benign, 1: malignant)
        self.data['label'] = self.data['filename'].apply(
            lambda x: 1 if 'malignant' in x else 0
        )
        
        # Set up transforms
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize(target_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transform
            
        logger.info(f"Loaded {len(self.data)} {split} images for fold {fold}, magnification {magnification}X")
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """Get a single image and its label"""
        row = self.data.iloc[idx]
        image_path = self.data_root / row['filename']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            image = torch.zeros(3, *self.target_size)
        
        label = row['label']
        filename = row['filename']
        
        return image, label, filename
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset"""
        labels = self.data['label'].values
        class_counts = np.bincount(labels)
        total_samples = len(labels)
        
        weights = total_samples / (len(class_counts) * class_counts)
        return torch.FloatTensor(weights)

class BreaKHisDataLoader:
    """Data loader manager for BreaKHis dataset"""
    
    def __init__(self, 
                 data_root: str,
                 folds_csv: str,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 target_size: Tuple[int, int] = (224, 224)):
        """
        Initialize data loader manager
        
        Args:
            data_root: Root directory containing BreaKHis_v1 folder
            folds_csv: Path to Folds.csv file
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes
            target_size: Target image size
        """
        self.data_root = data_root
        self.folds_csv = folds_csv
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_size = target_size
        
        # Define transforms for training and validation
        self.train_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def get_data_loaders(self, 
                        fold: int = 1,
                        magnification: str = '100') -> Tuple[DataLoader, DataLoader]:
        """
        Get train and validation data loaders for a specific fold
        
        Args:
            fold: Fold number (1-5)
            magnification: Magnification level ('40', '100', '200', '400')
            
        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create datasets
        train_dataset = BreaKHisDataset(
            data_root=self.data_root,
            folds_csv=self.folds_csv,
            fold=fold,
            split='train',
            magnification=magnification,
            transform=self.train_transform,
            target_size=self.target_size
        )
        
        val_dataset = BreaKHisDataset(
            data_root=self.data_root,
            folds_csv=self.folds_csv,
            fold=fold,
            split='test',
            magnification=magnification,
            transform=self.val_transform,
            target_size=self.target_size
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        return train_loader, val_loader
    
    def get_dataset_info(self) -> Dict:
        """Get information about the dataset"""
        folds_df = pd.read_csv(self.folds_csv)
        
        info = {
            'total_images': len(folds_df),
            'folds': sorted(folds_df['fold'].unique()),
            'magnifications': sorted(folds_df['mag'].unique()),
            'splits': sorted(folds_df['grp'].unique()),
            'benign_count': len(folds_df[folds_df['filename'].str.contains('benign')]),
            'malignant_count': len(folds_df[folds_df['filename'].str.contains('malignant')])
        }
        
        return info
    
    def get_sample_images(self, 
                         fold: int = 1,
                         magnification: str = '100',
                         num_samples: int = 5) -> List[Tuple[Image.Image, int, str]]:
        """
        Get sample images for visualization
        
        Args:
            fold: Fold number
            magnification: Magnification level
            num_samples: Number of samples to return
            
        Returns:
            List of (image, label, filename) tuples
        """
        dataset = BreaKHisDataset(
            data_root=self.data_root,
            folds_csv=self.folds_csv,
            fold=fold,
            split='train',
            magnification=magnification,
            transform=None,  # No transform for visualization
            target_size=self.target_size
        )
        
        samples = []
        for i in range(min(num_samples, len(dataset))):
            image, label, filename = dataset[i]
            samples.append((image, label, filename))
        
        return samples 