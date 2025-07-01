"""
BreaKHis Dataset Explorer Component
Provides interface for exploring and working with BreaKHis dataset
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import sys
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from utils.dataset_loader import BreaKHisDataLoader, BreaKHisDataset

class DatasetExplorer:
    """Component for exploring BreaKHis dataset"""
    
    def __init__(self, config):
        self.config = config
        self.dataset_config = config.get('dataset', {})
        
        # Initialize data loader
        self.data_loader = BreaKHisDataLoader(
            data_root=self.dataset_config.get('data_root', './BreaKHis_v1'),
            folds_csv=self.dataset_config.get('folds_csv', './Folds.csv'),
            batch_size=self.dataset_config.get('preprocessing', {}).get('batch_size', 32),
            num_workers=self.dataset_config.get('preprocessing', {}).get('num_workers', 4),
            target_size=tuple(self.dataset_config.get('preprocessing', {}).get('target_size', [224, 224]))
        )
    
    def render(self):
        """Render the dataset explorer interface"""
        st.header("🔬 BreaKHis Dataset Explorer")
        
        # Dataset overview
        self._show_dataset_overview()
        
        # Dataset configuration
        self._show_dataset_config()
        
        # Sample images
        self._show_sample_images()
        
        # Dataset statistics
        self._show_dataset_statistics()
    
    def _show_dataset_overview(self):
        """Show dataset overview information"""
        st.subheader("📊 Dataset Overview")
        
        try:
            dataset_info = self.data_loader.get_dataset_info()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Images", dataset_info['total_images'])
            
            with col2:
                st.metric("Benign Images", dataset_info['benign_count'])
            
            with col3:
                st.metric("Malignant Images", dataset_info['malignant_count'])
            
            with col4:
                malignant_ratio = dataset_info['malignant_count'] / dataset_info['total_images'] * 100
                st.metric("Malignant Ratio", f"{malignant_ratio:.1f}%")
            
            # Show available options
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Available Magnifications:**")
                for mag in dataset_info['magnifications']:
                    st.write(f"• {mag}X")
            
            with col2:
                st.write("**Available Folds:**")
                for fold in dataset_info['folds']:
                    st.write(f"• Fold {fold}")
                    
        except Exception as e:
            st.error(f"Error loading dataset info: {e}")
    
    def _show_dataset_config(self):
        """Show dataset configuration options"""
        st.subheader("⚙️ Dataset Configuration")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_fold = st.selectbox(
                "Select Fold",
                options=self.dataset_config.get('available_folds', [1, 2, 3, 4, 5]),
                index=0,
                help="Choose the fold for cross-validation"
            )
        
        with col2:
            selected_magnification = st.selectbox(
                "Select Magnification",
                options=self.dataset_config.get('available_magnifications', ["40", "100", "200", "400"]),
                index=1,  # Default to 100X
                help="Choose the magnification level"
            )
        
        with col3:
            batch_size = st.selectbox(
                "Batch Size",
                options=[16, 32, 64, 128],
                index=1,  # Default to 32
                help="Batch size for data loading"
            )
        
        # Store selections in session state
        st.session_state.selected_fold = selected_fold
        st.session_state.selected_magnification = selected_magnification
        st.session_state.batch_size = batch_size
        
        # Load dataset button
        if st.button("🔄 Load Dataset", type="primary"):
            with st.spinner("Loading dataset..."):
                try:
                    train_loader, val_loader = self.data_loader.get_data_loaders(
                        fold=selected_fold,
                        magnification=selected_magnification
                    )
                    
                    st.session_state.train_loader = train_loader
                    st.session_state.val_loader = val_loader
                    
                    st.success(f"✅ Dataset loaded successfully!")
                    try:
                        train_samples = len(train_loader.dataset)
                        val_samples = len(val_loader.dataset)
                        st.info(f"Train samples: {train_samples}, Validation samples: {val_samples}")
                    except:
                        st.info("Dataset loaded successfully!")
                    
                except Exception as e:
                    st.error(f"Error loading dataset: {e}")
    
    def _show_sample_images(self):
        """Show sample images from the dataset"""
        st.subheader("🖼️ Sample Images")
        
        if 'selected_fold' not in st.session_state:
            st.warning("Please configure and load the dataset first.")
            return
        
        try:
            samples = self.data_loader.get_sample_images(
                fold=st.session_state.selected_fold,
                magnification=st.session_state.selected_magnification,
                num_samples=6
            )
            
            # Display samples in a grid
            cols = st.columns(3)
            for i, (image, label, filename) in enumerate(samples):
                with cols[i % 3]:
                    st.image(image, caption=f"{'Malignant' if label else 'Benign'}", use_container_width=True)
                    st.caption(f"File: {Path(filename).name}")
                    
        except Exception as e:
            st.error(f"Error loading sample images: {e}")
    
    def _show_dataset_statistics(self):
        """Show detailed dataset statistics"""
        st.subheader("📈 Dataset Statistics")
        
        try:
            # Load folds data for analysis
            folds_df = pd.read_csv(self.dataset_config.get('folds_csv', './Folds.csv'))
            
            # Create labels
            folds_df['label'] = folds_df['filename'].apply(
                lambda x: 'Malignant' if 'malignant' in x else 'Benign'
            )
            
            # Distribution by magnification
            fig_mag = px.histogram(
                folds_df, 
                x='mag', 
                color='label',
                title="Distribution by Magnification",
                labels={'mag': 'Magnification', 'label': 'Class'},
                barmode='group'
            )
            st.plotly_chart(fig_mag, use_container_width=True)
            
            # Distribution by fold
            fig_fold = px.histogram(
                folds_df, 
                x='fold', 
                color='label',
                title="Distribution by Fold",
                labels={'fold': 'Fold', 'label': 'Class'},
                barmode='group'
            )
            st.plotly_chart(fig_fold, use_container_width=True)
            
            # Class balance
            class_counts = folds_df['label'].value_counts()
            fig_pie = px.pie(
                values=class_counts.values,
                names=class_counts.index,
                title="Class Distribution"
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error generating statistics: {e}")
    
    def get_current_dataset(self):
        """Get the currently loaded dataset"""
        if 'train_loader' in st.session_state and 'val_loader' in st.session_state:
            return st.session_state.train_loader, st.session_state.val_loader
        return None, None 