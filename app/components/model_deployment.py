"""
Model Deployment Component for PathoDetect+
Provides model deployment, versioning, and API management features
"""

import streamlit as st
import os
import json
import shutil
from pathlib import Path
import sys
import subprocess
import requests
from datetime import datetime
import yaml
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

class ModelDeployment:
    """Model deployment and management component"""
    
    def __init__(self, config):
        self.config = config
        self.models_dir = Path("models/saved")
        self.deployment_dir = Path("deployment")
        self.api_dir = Path("api")
        
        # Create directories if they don't exist
        self.deployment_dir.mkdir(exist_ok=True)
        self.api_dir.mkdir(exist_ok=True)
    
    def render(self):
        """Render the model deployment interface"""
        st.header("🚀 Model Deployment")
        
        # Deployment tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📦 Model Management", 
            "🌐 API Deployment", 
            "📊 Model Monitoring", 
            "🔧 Configuration"
        ])
        
        with tab1:
            self._show_model_management()
        
        with tab2:
            self._show_api_deployment()
        
        with tab3:
            self._show_model_monitoring()
        
        with tab4:
            self._show_deployment_config()
    
    def _show_model_management(self):
        """Show model management interface"""
        st.subheader("📦 Model Management")
        
        # List available models
        if self.models_dir.exists():
            model_files = [f for f in os.listdir(self.models_dir) if f.endswith('.pth')]
        else:
            model_files = []
        
        if not model_files:
            st.warning("No trained models found. Train a model first.")
            return
        
        # Model selection and deployment
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_model = st.selectbox(
                "Select Model to Deploy",
                model_files,
                help="Choose a trained model for deployment"
            )
            
            # Model information
            if selected_model:
                model_path = self.models_dir / selected_model
                model_info = self._get_model_info(model_path)
                
                st.write("**Model Information:**")
                st.json(model_info)
        
        with col2:
            # Deployment actions
            st.write("**Deployment Actions:**")
            
            if st.button("📦 Package Model", type="primary"):
                self._package_model(selected_model)
            
            if st.button("🚀 Deploy to Production"):
                self._deploy_to_production(selected_model)
            
            if st.button("🔄 Create New Version"):
                self._create_new_version(selected_model)
        
        # Model versions
        st.subheader("📋 Model Versions")
        
        versions = self._get_model_versions()
        if versions:
            version_df = pd.DataFrame(versions)
            st.dataframe(version_df, use_container_width=True)
        else:
            st.info("No deployed model versions found.")
        
        # Model registry
        st.subheader("🏪 Model Registry")
        
        registry_models = self._get_registry_models()
        if registry_models:
            for model in registry_models:
                with st.expander(f"📦 {model['name']} - v{model['version']}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Status:** {model['status']}")
                        st.write(f"**Created:** {model['created']}")
                        st.write(f"**Size:** {model['size']}")
                    
                    with col2:
                        st.write(f"**Accuracy:** {model['accuracy']:.2f}%")
                        st.write(f"**Environment:** {model['environment']}")
                        
                        if st.button(f"🔄 Rollback to v{model['version']}", key=f"rollback_{model['version']}"):
                            self._rollback_model(model)
        else:
            st.info("No models in registry.")
    
    def _get_model_info(self, model_path):
        """Get information about a model"""
        try:
            # Load model checkpoint
            import torch
            checkpoint = torch.load(model_path, map_location='cpu')
            
            info = {
                'filename': model_path.name,
                'size_mb': round(model_path.stat().st_size / (1024 * 1024), 2),
                'created': datetime.fromtimestamp(model_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'epoch': checkpoint.get('epoch', 'Unknown'),
                'best_val_acc': checkpoint.get('best_val_acc', 'Unknown'),
                'config': checkpoint.get('config', {})
            }
            
            return info
        except Exception as e:
            return {'error': str(e)}
    
    def _package_model(self, model_name):
        """Package a model for deployment"""
        try:
            with st.spinner("Packaging model..."):
                # Create deployment package
                package_name = f"model_package_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                package_dir = self.deployment_dir / package_name
                package_dir.mkdir(exist_ok=True)
                
                # Copy model file
                model_path = self.models_dir / model_name
                shutil.copy2(model_path, package_dir / model_name)
                
                # Create deployment config
                deployment_config = {
                    'model_name': model_name,
                    'version': '1.0.0',
                    'deployment_date': datetime.now().isoformat(),
                    'environment': 'production',
                    'requirements': [
                        'torch>=2.0.0',
                        'torchvision>=0.15.0',
                        'numpy>=1.24.0',
                        'pillow>=10.0.0'
                    ]
                }
                
                with open(package_dir / 'deployment_config.json', 'w') as f:
                    json.dump(deployment_config, f, indent=2)
                
                # Create README
                readme_content = f"""
# Model Deployment Package

**Model:** {model_name}
**Version:** 1.0.0
**Deployment Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Usage

1. Install requirements: `pip install -r requirements.txt`
2. Load model: `torch.load('{model_name}')`
3. Run inference: Use the provided inference script

## Configuration

See `deployment_config.json` for detailed configuration.
                """
                
                with open(package_dir / 'README.md', 'w') as f:
                    f.write(readme_content)
                
                st.success(f"✅ Model packaged successfully: {package_name}")
                
        except Exception as e:
            st.error(f"❌ Error packaging model: {e}")
    
    def _deploy_to_production(self, model_name):
        """Deploy model to production"""
        try:
            with st.spinner("Deploying to production..."):
                # Create production deployment
                prod_dir = self.deployment_dir / "production"
                prod_dir.mkdir(exist_ok=True)
                
                # Copy model to production
                model_path = self.models_dir / model_name
                prod_model_path = prod_dir / model_name
                shutil.copy2(model_path, prod_model_path)
                
                # Update production config
                prod_config = {
                    'active_model': model_name,
                    'deployment_date': datetime.now().isoformat(),
                    'status': 'active',
                    'version': '1.0.0'
                }
                
                with open(prod_dir / 'production_config.json', 'w') as f:
                    json.dump(prod_config, f, indent=2)
                
                st.success("✅ Model deployed to production successfully!")
                
        except Exception as e:
            st.error(f"❌ Error deploying to production: {e}")
    
    def _create_new_version(self, model_name):
        """Create a new version of a model"""
        try:
            # Get current version
            current_version = "1.0.0"  # This would be extracted from model metadata
            
            # Increment version
            version_parts = current_version.split('.')
            new_patch = str(int(version_parts[2]) + 1)
            new_version = f"{version_parts[0]}.{version_parts[1]}.{new_patch}"
            
            # Create new version directory
            version_dir = self.deployment_dir / f"v{new_version}"
            version_dir.mkdir(exist_ok=True)
            
            # Copy model with new version
            model_path = self.models_dir / model_name
            new_model_name = model_name.replace('.pth', f'_v{new_version}.pth')
            shutil.copy2(model_path, version_dir / new_model_name)
            
            st.success(f"✅ New version created: v{new_version}")
            
        except Exception as e:
            st.error(f"❌ Error creating new version: {e}")
    
    def _get_model_versions(self):
        """Get list of model versions"""
        versions = []
        
        if self.deployment_dir.exists():
            for item in self.deployment_dir.iterdir():
                if item.is_dir() and item.name.startswith('v'):
                    version_info = {
                        'version': item.name,
                        'created': datetime.fromtimestamp(item.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                        'models': len(list(item.glob('*.pth')))
                    }
                    versions.append(version_info)
        
        return versions
    
    def _get_registry_models(self):
        """Get models from registry"""
        # Simulated registry data
        return [
            {
                'name': 'ResNet-50',
                'version': '1.0.0',
                'status': 'active',
                'created': '2024-01-15 10:30:00',
                'size': '45.2 MB',
                'accuracy': 87.3,
                'environment': 'production'
            },
            {
                'name': 'EfficientNet-B4',
                'version': '1.1.0',
                'status': 'staging',
                'created': '2024-01-20 14:15:00',
                'size': '38.7 MB',
                'accuracy': 89.1,
                'environment': 'staging'
            }
        ]
    
    def _rollback_model(self, model):
        """Rollback to a specific model version"""
        st.info(f"Rolling back to {model['name']} v{model['version']}...")
        # Implementation would handle the rollback process
    
    def _show_api_deployment(self):
        """Show API deployment interface"""
        st.subheader("🌐 API Deployment")
        
        # API status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("API Status", "🟢 Running", "Production")
        
        with col2:
            st.metric("Requests/min", "1,247", "12%")
        
        with col3:
            st.metric("Response Time", "45ms", "-5ms")
        
        # API endpoints
        st.subheader("📡 API Endpoints")
        
        endpoints = [
            {
                'endpoint': '/api/v1/predict',
                'method': 'POST',
                'description': 'Predict cancer probability for image',
                'status': 'active'
            },
            {
                'endpoint': '/api/v1/health',
                'method': 'GET',
                'description': 'Health check endpoint',
                'status': 'active'
            },
            {
                'endpoint': '/api/v1/models',
                'method': 'GET',
                'description': 'List available models',
                'status': 'active'
            }
        ]
        
        for endpoint in endpoints:
            with st.expander(f"{endpoint['method']} {endpoint['endpoint']}"):
                st.write(f"**Description:** {endpoint['description']}")
                st.write(f"**Status:** {endpoint['status']}")
                
                if endpoint['endpoint'] == '/api/v1/predict':
                    st.code("""
curl -X POST "http://localhost:8000/api/v1/predict" \\
  -H "Content-Type: multipart/form-data" \\
  -F "image=@path/to/image.jpg"
                    """, language='bash')
        
        # API deployment controls
        st.subheader("🔧 API Controls")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🚀 Start API Server"):
                self._start_api_server()
        
        with col2:
            if st.button("⏸️ Pause API"):
                self._pause_api()
        
        with col3:
            if st.button("🛑 Stop API Server"):
                self._stop_api_server()
        
        # API logs
        st.subheader("📋 API Logs")
        
        # Simulated logs
        logs = [
            "2024-01-25 10:30:15 - INFO - API server started on port 8000",
            "2024-01-25 10:30:20 - INFO - Model loaded successfully",
            "2024-01-25 10:31:05 - INFO - Prediction request received",
            "2024-01-25 10:31:08 - INFO - Prediction completed in 45ms",
            "2024-01-25 10:32:12 - WARNING - High memory usage detected"
        ]
        
        for log in logs:
            st.text(log)
    
    def _start_api_server(self):
        """Start the API server"""
        st.info("Starting API server...")
        # Implementation would start the FastAPI server
    
    def _pause_api(self):
        """Pause the API server"""
        st.info("Pausing API server...")
        # Implementation would pause the API
    
    def _stop_api_server(self):
        """Stop the API server"""
        st.info("Stopping API server...")
        # Implementation would stop the API
    
    def _show_model_monitoring(self):
        """Show model monitoring interface"""
        st.subheader("📊 Model Monitoring")
        
        # Real-time metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Active Requests", "23", "5")
        
        with col2:
            st.metric("Success Rate", "98.5%", "0.2%")
        
        with col3:
            st.metric("Error Rate", "1.5%", "-0.2%")
        
        with col4:
            st.metric("Model Load", "67%", "3%")
        
        # Performance metrics over time
        st.subheader("Performance Over Time")
        
        # Generate sample performance data
        import plotly.graph_objects as go
        import numpy as np
        
        hours = list(range(24))
        requests_per_hour = [np.random.randint(50, 200) for _ in hours]
        response_times = [np.random.uniform(30, 80) for _ in hours]
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hours, y=requests_per_hour, name='Requests/Hour', yaxis='y'))
        fig.add_trace(go.Scatter(x=hours, y=response_times, name='Response Time (ms)', yaxis='y2'))
        
        fig.update_layout(
            title="API Performance (24 Hours)",
            xaxis_title="Hour",
            yaxis=dict(title="Requests/Hour", side='left'),
            yaxis2=dict(title="Response Time (ms)", side='right', overlaying='y')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Error tracking
        st.subheader("Error Tracking")
        
        errors = [
            {'timestamp': '10:30:15', 'error': 'Invalid image format', 'count': 3},
            {'timestamp': '10:31:22', 'error': 'Model timeout', 'count': 1},
            {'timestamp': '10:32:08', 'error': 'Memory allocation failed', 'count': 2}
        ]
        
        for error in errors:
            with st.expander(f"❌ {error['error']} ({error['count']} occurrences)"):
                st.write(f"**Time:** {error['timestamp']}")
                st.write(f"**Count:** {error['count']}")
                st.write("**Resolution:** Check input format and system resources")
    
    def _show_deployment_config(self):
        """Show deployment configuration"""
        st.subheader("🔧 Deployment Configuration")
        
        # Environment configuration
        st.subheader("Environment Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            api_host = st.text_input("API Host", value="0.0.0.0")
            api_port = st.number_input("API Port", value=8000, min_value=1000, max_value=9999)
            workers = st.number_input("Number of Workers", value=4, min_value=1, max_value=16)
        
        with col2:
            timeout = st.number_input("Request Timeout (s)", value=30, min_value=5, max_value=300)
            max_file_size = st.number_input("Max File Size (MB)", value=50, min_value=1, max_value=500)
            log_level = st.selectbox("Log Level", ["INFO", "DEBUG", "WARNING", "ERROR"])
        
        # Model configuration
        st.subheader("Model Settings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            batch_size = st.number_input("Batch Size", value=32, min_value=1, max_value=128)
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.5, 0.1)
        
        with col2:
            enable_caching = st.checkbox("Enable Model Caching", value=True)
            enable_monitoring = st.checkbox("Enable Monitoring", value=True)
        
        # Save configuration
        if st.button("💾 Save Configuration"):
            config = {
                'api': {
                    'host': api_host,
                    'port': api_port,
                    'workers': workers,
                    'timeout': timeout,
                    'max_file_size': max_file_size,
                    'log_level': log_level
                },
                'model': {
                    'batch_size': batch_size,
                    'confidence_threshold': confidence_threshold,
                    'enable_caching': enable_caching,
                    'enable_monitoring': enable_monitoring
                }
            }
            
            with open(self.deployment_dir / 'deployment_config.yaml', 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            st.success("✅ Configuration saved successfully!") 