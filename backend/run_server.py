#!/usr/bin/env python3
"""
Startup script for the Medical Image Prediction API
This script handles the correct setup and running of the FastAPI server
"""

import sys
import os
from pathlib import Path
import subprocess
from dotenv import load_dotenv

def setup_python_path():
    """Add necessary directories to Python path"""
    current_dir = Path(__file__).resolve().parent
    app_dir = current_dir / "app"
    backend_dir = current_dir
    
    # Add paths to sys.path
    sys.path.insert(0, str(app_dir))
    sys.path.insert(0, str(backend_dir))
    
    print(f"Added to Python path:")
    print(f"  - {app_dir}")
    print(f"  - {backend_dir}")
    
    return app_dir

def check_requirements():
    """Check if required packages are installed using correct import names"""
    required_packages = {
        'fastapi': 'fastapi',
        'uvicorn': 'uvicorn',
        'tensorflow': 'tensorflow',
        'pillow': 'PIL',          
        'opencv-python': 'cv2',   
        'scikit-image': 'skimage',# Actual import name
        'numpy': 'numpy',
        'scipy': 'scipy'
    }
    
    missing_packages = []
    for pip_name, import_name in required_packages.items():
        try:
            __import__(import_name)  # Use the actual import name here
        except ImportError:
            missing_packages.append(pip_name)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install " + " ".join(missing_packages))
        return False
    
    print("✓ All required packages are installed")
    return True

def run_server(port=8001, host="127.0.0.1", reload=True):
    """Run the FastAPI server"""
    # load environment variables from .env file
    load_dotenv()       
    
    app_dir = setup_python_path()
    
    
    if not check_requirements():
        print("Please install missing packages before running the server")
        return False
    
    # Change to the app directory
    os.chdir(app_dir)
    
    # Run uvicorn
    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"Starting server on http://{host}:{port}")
    print(f"Command: {' '.join(cmd)}")
    print("Press Ctrl+C to stop the server")
    print("-" * 50)
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"Error running server: {e}")
        return False
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the Medical Image Prediction API")
    parser.add_argument("--port", "-p", type=int, default=8001, help="Port to run the server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    
    args = parser.parse_args()
    
    run_server(
        port=args.port,
        host=args.host,
        reload=not args.no_reload
    )