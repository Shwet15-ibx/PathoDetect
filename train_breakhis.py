#!/usr/bin/env python3
"""
BreaKHis Training Script for PathoDetect+
Trains models on the BreaKHis dataset for breast cancer classification
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.dataset_loader import BreaKHisDataLoader
from models.resnet_classifier import ResNetClassifier
from models.efficientnet_classifier import EfficientNetClassifier
from mlops.tracking import MLflowTracker

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BreaKHisTrainer:
    """Trainer class for BreaKHis dataset"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize trainer with configuration"""
        self.config = self._load_config(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize data loader
        self.data_loader = BreaKHisDataLoader(
            data_root=self.config['dataset']['data_root'],
            folds_csv=self.config['dataset']['folds_csv'],
            batch_size=self.config['training']['batch_size'],
            num_workers=self.config['dataset']['preprocessing']['num_workers'],
            target_size=tuple(self.config['dataset']['preprocessing']['target_size'])
        )
        
        # Initialize MLflow tracker
        self.tracker = MLflowTracker(self.config)
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        with open(config_path, 'r') as file:
            return yaml.safe_load(file)
    
    def _create_model(self, model_name: str) -> nn.Module:
        """Create model based on name"""
        if model_name == "resnet":
            model = ResNetClassifier(
                num_classes=self.config['models']['resnet']['num_classes'],
                dropout_rate=self.config['models']['resnet']['dropout_rate']
            )
        elif model_name == "efficientnet":
            model = EfficientNetClassifier(
                num_classes=self.config['models']['efficientnet']['num_classes'],
                dropout_rate=self.config['models']['efficientnet']['dropout_rate']
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model.to(self.device)
    
    def _create_optimizer(self, model: nn.Module, model_name: str) -> optim.Optimizer:
        """Create optimizer for model"""
        if model_name == "resnet":
            lr = self.config['models']['resnet']['learning_rate']
        elif model_name == "efficientnet":
            lr = self.config['models']['efficientnet']['learning_rate']
        else:
            lr = 0.001
        
        return optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=self.config['training']['weight_decay']
        )
    
    def _create_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler"""
        scheduler_type = self.config['training']['learning_rate_scheduler']
        if scheduler_type == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['training']['num_epochs']
            )
        else:
            return optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    def train_model(self, 
                   model_name: str = "resnet",
                   fold: int = 1,
                   magnification: str = "100") -> dict:
        """
        Train a model on BreaKHis dataset
        
        Args:
            model_name: Name of the model to train
            fold: Fold number for cross-validation
            magnification: Magnification level
            
        Returns:
            Dictionary containing training results
        """
        logger.info(f"Starting training for {model_name} on fold {fold}, magnification {magnification}X")
        
        # Get data loaders
        train_loader, val_loader = self.data_loader.get_data_loaders(fold, magnification)
        
        # Create model
        model = self._create_model(model_name)
        
        # Create optimizer and scheduler
        optimizer = self._create_optimizer(model, model_name)
        scheduler = self._create_scheduler(optimizer)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        training_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(self.config['training']['num_epochs']):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - Training")
            for batch_idx, (images, labels, _) in enumerate(train_pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['gradient_clipping'] > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config['training']['gradient_clipping'])
                
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = outputs.max(1)
                train_total += labels.size(0)
                train_correct += predicted.eq(labels).sum().item()
                
                train_pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100.*train_correct/train_total:.2f}%"
                })
            
            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_predictions = []
            val_labels = []
            
            with torch.no_grad():
                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{self.config['training']['num_epochs']} - Validation")
                for images, labels, _ in val_pbar:
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()
                    
                    val_predictions.extend(outputs.softmax(1)[:, 1].cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
                    
                    val_pbar.set_postfix({
                        'Loss': f"{loss.item():.4f}",
                        'Acc': f"{100.*val_correct/val_total:.2f}%"
                    })
            
            # Calculate metrics
            train_acc = 100. * train_correct / train_total
            val_acc = 100. * val_correct / val_total
            
            # Calculate additional metrics
            val_auc = roc_auc_score(val_labels, val_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                val_labels, 
                [1 if p > 0.5 else 0 for p in val_predictions], 
                average='binary'
            )
            
            # Update history
            training_history['train_loss'].append(train_loss / len(train_loader))
            training_history['train_acc'].append(train_acc)
            training_history['val_loss'].append(val_loss / len(val_loader))
            training_history['val_acc'].append(val_acc)
            
            # Log metrics
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss / len(train_loader),
                'train_acc': train_acc,
                'val_loss': val_loss / len(val_loader),
                'val_acc': val_acc,
                'val_auc': val_auc,
                'val_precision': precision,
                'val_recall': recall,
                'val_f1': f1
            }
            
            self.tracker.log_metrics(metrics)
            
            logger.info(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%, Val AUC: {val_auc:.4f}")
            
            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                
                # Save best model
                model_path = f"models/saved/{model_name}_fold{fold}_mag{magnification}_best.pth"
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_acc': best_val_acc,
                    'config': self.config
                }, model_path)
                logger.info(f"Saved best model to {model_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config['training']['early_stopping_patience']:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
            
            scheduler.step()
        
        # Final evaluation
        final_metrics = {
            'best_val_acc': best_val_acc,
            'final_epoch': epoch + 1,
            'model_name': model_name,
            'fold': fold,
            'magnification': magnification
        }
        
        self.tracker.log_metrics(final_metrics)
        
        return {
            'model_name': model_name,
            'fold': fold,
            'magnification': magnification,
            'best_val_acc': best_val_acc,
            'training_history': training_history,
            'final_metrics': final_metrics
        }
    
    def train_all_models(self, magnifications: list = ["100"]) -> dict:
        """
        Train all models on all folds and magnifications
        
        Args:
            magnifications: List of magnifications to train on
            
        Returns:
            Dictionary containing all training results
        """
        results = {}
        
        for magnification in magnifications:
            for fold in self.config['dataset']['available_folds']:
                for model_name in ["resnet", "efficientnet"]:
                    logger.info(f"Training {model_name} on fold {fold}, magnification {magnification}X")
                    
                    try:
                        result = self.train_model(model_name, fold, magnification)
                        results[f"{model_name}_fold{fold}_mag{magnification}"] = result
                    except Exception as e:
                        logger.error(f"Error training {model_name} on fold {fold}, magnification {magnification}: {e}")
                        results[f"{model_name}_fold{fold}_mag{magnification}"] = {'error': str(e)}
        
        return results

def main():
    """Main training function"""
    trainer = BreaKHisTrainer()
    
    # Train on 100X magnification for demonstration
    results = trainer.train_all_models(magnifications=["100"])
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    for key, result in results.items():
        if 'error' not in result:
            print(f"{key}: Best Val Acc: {result['best_val_acc']:.2f}%")
        else:
            print(f"{key}: Error - {result['error']}")

if __name__ == "__main__":
    main() 