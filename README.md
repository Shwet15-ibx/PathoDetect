# PathoDetect+: Histopathology Image Classifier with LLM-Assisted Reporting

🔍 **Advanced histopathology image analysis with AI-powered cancer detection and explainable reporting**

## 🚀 Features

- **High-Resolution Slide Processing**: Upload and analyze pathology slides with patch-based classification
- **AI-Powered Detection**: ResNet and EfficientNet models for cancerous tissue detection
- **Interactive Visualization**: Streamlit-based patch viewer with cancer heatmap overlay
- **LLM-Assisted Reporting**: LangChain integration for Q&A and summary explanations
- **MLOps Integration**: Track F1, AUC, and overlay visualization accuracy
- **Explainable AI**: Generate detailed reports for clinicians

## 📁 Project Structure

```
PathoDetect/
├── app/
│   ├── main.py                 # Main Streamlit application
│   ├── components/
│   │   ├── image_uploader.py   # Image upload component
│   │   ├── patch_viewer.py     # Patch visualization component
│   │   ├── heatmap_overlay.py  # Heatmap overlay component
│   │   └── report_generator.py # LLM report generation
│   └── utils/
│       ├── image_processing.py # Image preprocessing utilities
│       ├── model_utils.py      # Model loading and inference
│       └── llm_utils.py        # LangChain integration
├── models/
│   ├── resnet_classifier.py    # ResNet-based classifier
│   ├── efficientnet_classifier.py # EfficientNet classifier
│   └── ensemble_model.py       # Ensemble of multiple models
├── data/
│   ├── raw/                    # Raw pathology images
│   ├── processed/              # Processed patches
│   └── annotations/            # Ground truth annotations
├── mlops/
│   ├── tracking.py             # MLflow tracking
│   ├── metrics.py              # Performance metrics
│   └── visualization.py        # Visualization accuracy tracking
├── config/
│   └── config.yaml             # Configuration settings
├── requirements.txt            # Python dependencies
├── .env.example               # Environment variables template
└── README.md                  # This file
```

## 🛠️ Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd PathoDetect
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

5. **Run the application**
```bash
streamlit run app/main.py
```

## 🔧 Configuration

Create a `.env` file with the following variables:
```
LANGCHAIN_API_KEY=your_langchain_api_key
GROQ_API_KEY=your_groq_api_key
MLFLOW_TRACKING_URI=your_mlflow_uri
```

## 📊 Usage

1. **Upload Pathology Slide**: Use the upload interface to load high-resolution pathology images
2. **Patch Analysis**: The system automatically divides the image into patches for analysis
3. **Cancer Detection**: AI models classify each patch for cancerous tissue
4. **Visualization**: View results with interactive heatmap overlay
5. **Report Generation**: Generate LLM-assisted reports with explanations

## 🤖 Models

- **ResNet-50**: Pre-trained on ImageNet, fine-tuned for histopathology
- **EfficientNet-B4**: Efficient architecture for patch classification
- **Ensemble Model**: Combines predictions from multiple models

## 📈 MLOps Features

- **Performance Tracking**: F1 score, AUC, precision, recall
- **Visualization Accuracy**: Track overlay accuracy metrics
- **Model Versioning**: MLflow integration for experiment tracking
- **A/B Testing**: Compare different model configurations

## 🔬 Technical Details

- **Patch Size**: 224x224 pixels (configurable)
- **Overlap**: 50% overlap between patches
- **Models**: PyTorch-based implementations
- **Frontend**: Streamlit for interactive interface
- **LLM**: LangChain with Groq API for report generation

## 📝 License

MIT License - see LICENSE file for details

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📞 Support

For questions and support, please open an issue on GitHub.

## Dataset

This project uses the [BreaKHis breast cancer histopathology image dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/).

**Note:** The dataset is not included in this repository due to size and copyright restrictions. Please download it from the official source above if you wish to use it. 