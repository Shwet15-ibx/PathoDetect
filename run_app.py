#!/usr/bin/env python3
"""
PathoDetect+ Startup Script
Run this script to start the Streamlit application
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the PathoDetect+ application"""
    
    # Check if we're in the right directory
    project_root = Path(__file__).parent
    app_path = project_root / "app" / "main.py"
    
    if not app_path.exists():
        print("❌ Error: app/main.py not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    print("🚀 Starting PathoDetect+ Application...")
    print("📁 Project root:", project_root)
    print("🔬 App path:", app_path)
    
    # Check if .env file exists
    env_file = project_root / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("   Create a .env file with your API keys:")
        print("   GROQ_API_KEY=your_groq_api_key_here")
        print("   LANGCHAIN_API_KEY=your_langchain_api_key_here")
    
    # Start Streamlit
    try:
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port", "8501",
            "--server.address", "localhost"
        ]
        
        print("🌐 Starting Streamlit server...")
        print("📱 Open your browser to: http://localhost:8501")
        print("⏹️  Press Ctrl+C to stop the server")
        
        subprocess.run(cmd, cwd=project_root)
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down PathoDetect+...")
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 