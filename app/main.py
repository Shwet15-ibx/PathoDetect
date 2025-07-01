"""
PathoDetect+: Main Streamlit Application
Histopathology Image Classifier with LLM-Assisted Reporting
"""

import streamlit as st
import os
import sys
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load environment variables
load_dotenv()

# Import custom modules
from components.image_uploader import ImageUploader
from components.patch_viewer import PatchViewer
from components.heatmap_overlay import HeatmapOverlay
from components.report_generator import ReportGenerator
from components.dataset_explorer import DatasetExplorer
from components.model_trainer import ModelTrainer
from components.advanced_analytics import AdvancedAnalytics
from components.model_deployment import ModelDeployment
from components.model_utils import initialize_models, get_model_status
from utils.image_processing import ImageProcessor
from utils.model_utils import ModelManager
from utils.llm_utils import LLMProcessor
from mlops.tracking import MLflowTracker

# Load configuration
def load_config():
    """Load configuration from YAML file"""
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state variables"""
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'processed_patches' not in st.session_state:
        st.session_state.processed_patches = None
    if 'predictions' not in st.session_state:
        st.session_state.predictions = None
    if 'heatmap' not in st.session_state:
        st.session_state.heatmap = None
    if 'report' not in st.session_state:
        st.session_state.report = None
    if 'model_manager' not in st.session_state:
        st.session_state.model_manager = None
    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = None

def main():
    """Main application function"""
    
    # Load configuration
    config = load_config()
    
    # Initialize session state
    init_session_state()
    
    # Initialize models and LLM at startup
    if 'model_manager' not in st.session_state or st.session_state.model_manager is None:
        print("🔄 Initializing models for the first time...")
        model_manager = initialize_models(config)
        if model_manager and model_manager.models:
            st.session_state.model_manager = model_manager
            print("✅ Models successfully loaded and stored in session state")
        else:
            print("❌ Failed to initialize models")
            st.session_state.model_manager = None
    else:
        # Check if models are actually loaded
        if not st.session_state.model_manager.models:
            print("🔄 Models exist but not loaded, loading now...")
            if st.session_state.model_manager.load_models():
                print("✅ Models loaded successfully at startup!")
            else:
                print("❌ Failed to load models at startup")
                st.session_state.model_manager = None
        else:
            print("✅ Models already loaded in session state")
    
    if 'llm_processor' not in st.session_state:
        try:
            llm_processor = LLMProcessor(config)
            st.session_state.llm_processor = llm_processor
        except Exception as e:
            st.session_state.llm_processor = None
            st.warning(f"LLM not connected: {e}")
    
    # Page configuration
    st.set_page_config(
        page_title=config['app']['title'],
        page_icon="🔬",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown(f"""
    <div class="main-header">
        <h1>{config['app']['title']}</h1>
        <p>{config['app']['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # Model selection
        model_option = st.selectbox(
            "Select Model",
            ["ResNet-50", "EfficientNet-B4", "Ensemble"],
            help="Choose the AI model for cancer detection"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.1,
            help="Minimum confidence for cancer detection"
        )
        
        # Patch size
        patch_size = st.selectbox(
            "Patch Size",
            [224, 256, 512],
            help="Size of image patches for analysis"
        )
        
        # Generate report option
        generate_report = st.checkbox(
            "Generate LLM Report",
            value=True,
            help="Generate AI-assisted analysis report"
        )
        
        # MLOps tracking
        enable_tracking = st.checkbox(
            "Enable MLOps Tracking",
            value=True,
            help="Track experiments and metrics"
        )
        
        st.divider()
        
        # System info
        st.subheader("📊 System Status")
        
        # Check model status
        models_loaded, current_model = get_model_status()
        if models_loaded:
            st.success(f"✅ Models Loaded ({current_model})")
        else:
            st.error("❌ Models Not Loaded")
            # Add debug info
            if st.button("🔄 Reload Models"):
                st.session_state.model_manager = None
                st.rerun()
            
        # Debug info
        with st.expander("🔍 Debug Info"):
            st.write(f"Model Manager exists: {st.session_state.get('model_manager') is not None}")
            model_manager = st.session_state.get('model_manager')
            if model_manager is not None:
                st.write(f"Models loaded: {len(model_manager.models)}")
                st.write(f"Available models: {list(model_manager.models.keys())}")
                st.write(f"Current model: {model_manager.current_model}")
            else:
                st.write("Model Manager is None")
            
        # Check LLM status
        llm_processor = st.session_state.get('llm_processor')
        if llm_processor:
            st.success("✅ LLM Connected")
        else:
            st.warning("⚠️ LLM Not Connected")
            # Debug info for LLM
            with st.expander("🔍 LLM Debug Info"):
                langchain_key = os.getenv('LANGCHAIN_API_KEY')
                groq_key = os.getenv('GROQ_API_KEY')
                st.write(f"LANGCHAIN_API_KEY found: {bool(langchain_key)}")
                st.write(f"GROQ_API_KEY found: {bool(groq_key)}")
                try:
                    import langchain_community
                    st.write("LangChain installed: True")
                except ImportError:
                    st.write("LangChain installed: False")
                st.write("If neither key is found, add them to your .env file and restart the app.")
    
    # Main content area
    # Create tabs for different functionalities
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🔬 Analysis", 
        "📊 Dataset Explorer", 
        "🤖 Model Trainer",
        "📈 Advanced Analytics",
        "🚀 Model Deployment",
        "📊 MLOps"
    ])
    
    with tab1:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📁 Image Upload & Analysis")
            
            # Image uploader
            uploader = ImageUploader(config)
            uploaded_file = uploader.upload_image()
            
            if uploaded_file is not None:
                # Process image
                with st.spinner("Processing image..."):
                    processor = ImageProcessor(config)
                    image_data = processor.load_image(uploaded_file)
                    if image_data:
                        patches = processor.extract_patches(image_data, patch_size)
                        st.session_state.processed_patches = patches
                        st.session_state.uploaded_image = image_data
                        
                        st.success(f"✅ Processed {len(patches)} patches")
                    else:
                        st.error("❌ Failed to load image")
        
        with col2:
            st.header("📈 Quick Stats")
            
            if st.session_state.processed_patches:
                st.metric("Total Patches", len(st.session_state.processed_patches))
                
            if st.session_state.predictions:
                cancer_patches = sum(1 for p in st.session_state.predictions if p['cancer_prob'] > confidence_threshold)
                st.metric("Cancer Patches", cancer_patches)
                st.metric("Cancer Rate", f"{cancer_patches/len(st.session_state.predictions)*100:.1f}%")
        
        # Analysis section
        if st.session_state.processed_patches:
            st.header("🔍 Analysis Results")
            
            # Initialize models if not already done
            if st.session_state.model_manager is None:
                model_manager = initialize_models(config)
            
            # Run predictions
            if st.button("🚀 Run Cancer Detection", type="primary"):
                if st.session_state.model_manager is None:
                    st.error("❌ Models not loaded. Please try again.")
                else:
                    with st.spinner("Running AI analysis..."):
                        predictions = st.session_state.model_manager.predict_patches(
                            st.session_state.processed_patches,
                            confidence_threshold
                        )
                        st.session_state.predictions = predictions
                        
                        # Generate heatmap
                        heatmap_generator = HeatmapOverlay(config)
                        heatmap = heatmap_generator.create_heatmap(
                            st.session_state.uploaded_image,
                            predictions,
                            patch_size
                        )
                        st.session_state.heatmap = heatmap
                        
                        st.success("✅ Analysis complete!")
            
            # Display results
            if st.session_state.predictions:
                # Results tabs
                result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs(["🎯 Heatmap", "🔬 Patches", "📊 Statistics", "📋 Report"])
                
                with result_tab1:
                    if st.session_state.heatmap:
                        st.subheader("Cancer Detection Heatmap")
                        st.image(st.session_state.heatmap, use_container_width=True)
                        
                        # Heatmap controls
                        col1, col2 = st.columns(2)
                        with col1:
                            opacity = st.slider("Heatmap Opacity", 0.1, 1.0, 0.7)
                        with col2:
                            colormap = st.selectbox("Color Map", ["RdYlBu_r", "Reds", "Blues", "Greens"])
                
                with result_tab2:
                    st.subheader("Patch Analysis")
                    patch_viewer = PatchViewer(config)
                    patch_viewer.display_patches(
                        st.session_state.processed_patches,
                        st.session_state.predictions
                    )
                
                with result_tab3:
                    st.subheader("Detection Statistics")
                    
                    # Calculate metrics
                    total_patches = len(st.session_state.predictions)
                    cancer_patches = sum(1 for p in st.session_state.predictions if p['cancer_prob'] > confidence_threshold)
                    avg_confidence = sum(p['cancer_prob'] for p in st.session_state.predictions) / total_patches
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Patches", total_patches)
                    with col2:
                        st.metric("Cancer Patches", cancer_patches)
                    with col3:
                        st.metric("Avg Confidence", f"{avg_confidence:.3f}")
                    
                    # Confidence distribution
                    import plotly.express as px
                    import pandas as pd
                    
                    confidences = [p['cancer_prob'] for p in st.session_state.predictions]
                    df = pd.DataFrame({'Confidence': confidences})
                    
                    fig = px.histogram(df, x='Confidence', nbins=20, 
                                     title="Confidence Distribution")
                    st.plotly_chart(fig, use_container_width=True)
                
                with result_tab4:
                    st.subheader("AI-Generated Report")
                    
                    if generate_report:
                        # Initialize LLM if not already done
                        if st.session_state.llm_processor is None:
                            with st.spinner("Connecting to LLM..."):
                                llm_processor = LLMProcessor(config)
                                st.session_state.llm_processor = llm_processor
                        
                        if st.button("📝 Generate Report"):
                            with st.spinner("Generating AI report..."):
                                report_generator = ReportGenerator(config)
                                report = report_generator.generate_report(
                                    st.session_state.predictions,
                                    st.session_state.uploaded_image,
                                    st.session_state.llm_processor
                                )
                                st.session_state.report = report
                                
                                # Display report
                                st.markdown(report)
                                
                                # Download button
                                st.download_button(
                                    label="📥 Download Report",
                                    data=report,
                                    file_name="pathology_report.md",
                                    mime="text/markdown"
                                )
                    else:
                        st.info("Enable 'Generate LLM Report' in sidebar to create AI-assisted reports")
    
    with tab2:
        # Dataset Explorer
        if config.get('app', {}).get('features', {}).get('enable_breakhis_dataset', False):
            dataset_explorer = DatasetExplorer(config)
            dataset_explorer.render()
        else:
            st.warning("BreaKHis dataset integration is not enabled in the configuration.")
    
    with tab3:
        # Model Trainer
        if config.get('app', {}).get('features', {}).get('enable_model_training', False):
            model_trainer = ModelTrainer(config)
            model_trainer.render()
        else:
            st.warning("Model training integration is not enabled in the configuration.")
    
    with tab4:
        # Advanced Analytics
        if config.get('app', {}).get('features', {}).get('enable_advanced_analytics', False):
            advanced_analytics = AdvancedAnalytics(config)
            advanced_analytics.render()
        else:
            st.warning("Advanced analytics integration is not enabled in the configuration.")
    
    with tab5:
        # Model Deployment
        if config.get('app', {}).get('features', {}).get('enable_model_deployment', False):
            model_deployment = ModelDeployment(config)
            model_deployment.render()
        else:
            st.warning("Model deployment integration is not enabled in the configuration.")
    
    with tab6:
        # MLOps tracking
        if enable_tracking and st.session_state.predictions:
            st.header("📊 MLOps Tracking")
            
            tracker = MLflowTracker(config)
            
            # Log metrics
            if st.button("📈 Log Experiment"):
                with st.spinner("Logging experiment..."):
                    metrics = {
                        'total_patches': len(st.session_state.predictions),
                        'cancer_patches': sum(1 for p in st.session_state.predictions if p['cancer_prob'] > confidence_threshold),
                        'avg_confidence': sum(p['cancer_prob'] for p in st.session_state.predictions) / len(st.session_state.predictions),
                        'model_used': model_option,
                        'confidence_threshold': confidence_threshold
                    }
                    
                    tracker.log_metrics(metrics)
                    tracker.log_artifacts({
                        'predictions': st.session_state.predictions,
                        'heatmap': st.session_state.heatmap
                    })
                    
                    st.success("✅ Experiment logged successfully!")
        else:
            st.info("Enable MLOps tracking and run analysis to log experiments.")

if __name__ == "__main__":
    main() 