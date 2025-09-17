"""
SOC Prediction System - Quick Start
Basit başlatma scripti
"""

import subprocess
import sys
import time
import os

def install_requirements():
    """Gerekli kütüphaneleri yükle"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def start_api():
    """API'yi başlat"""
    print("Starting API...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "src.api.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000",
        "--reload"
    ])

def start_demo():
    """Demo'yu başlat"""
    print("Starting demo...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", 
        "demo/streamlit_app.py",
        "--server.port", "8501"
    ])

def main():
    print("SOC Prediction System")
    print("=" * 30)
    
    try:
        # Requirements kontrol
        try:
            import tensorflow, fastapi, streamlit
        except ImportError:
            install_requirements()
        
        # API başlat
        api_process = start_api()
        time.sleep(5)
        
        # Demo başlat
        demo_process = start_demo()
        
        print("\nSystem started!")
        print("API: http://localhost:8000")
        print("Demo: http://localhost:8501")
        print("Press Ctrl+C to stop")
        
        # Bekle
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\nStopping...")
            api_process.terminate()
            demo_process.terminate()
    
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
