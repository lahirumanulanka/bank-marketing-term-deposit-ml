#!/usr/bin/env python3
"""
Startup script for Bank Marketing Term Deposit Prediction
Supports both Gradio and FastAPI modes
"""

import os
import sys
import argparse
import subprocess

def run_gradio():
    """Run the Gradio interface"""
    print("🚀 Starting Gradio interface...")
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Gradio interface stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Gradio: {e}")

def run_fastapi():
    """Run the FastAPI server"""
    print("🚀 Starting FastAPI server...")
    try:
        import uvicorn
        uvicorn.run(
            "api_app:app",
            host="0.0.0.0",
            port=7860,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n👋 FastAPI server stopped by user")
    except ImportError:
        print("❌ uvicorn not installed. Install with: pip install uvicorn")
    except Exception as e:
        print(f"❌ Error running FastAPI: {e}")

def check_model_files():
    """Check if required model files exist"""
    required_files = [
        "xgboost_retrained_tuned.pkl",
        "preprocessing/scaler.pkl",
        "preprocessing/label_encoders.pkl"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Warning: Missing model files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\nMake sure to copy the model files to this directory.")
        return False
    
    print("✅ All model files found")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Bank Marketing Term Deposit Prediction Server"
    )
    parser.add_argument(
        "--mode",
        choices=["gradio", "fastapi", "api"],
        default="gradio",
        help="Server mode (default: gradio)"
    )
    parser.add_argument(
        "--check-models",
        action="store_true",
        help="Check if model files exist"
    )
    
    args = parser.parse_args()
    
    print("🏦 Bank Marketing Term Deposit Prediction")
    print("=" * 50)
    
    # Check model files
    if args.check_models or not check_model_files():
        if args.check_models:
            return
        
        response = input("\nContinue anyway? (y/N): ").strip().lower()
        if response != 'y':
            print("Exiting...")
            return
    
    # Run selected mode
    if args.mode == "gradio":
        print("\n📱 Mode: Gradio Web Interface")
        print("   Access: http://localhost:7860")
        print("   Features: Interactive web form")
        run_gradio()
        
    elif args.mode in ["fastapi", "api"]:
        print("\n🔌 Mode: FastAPI REST API")
        print("   Access: http://localhost:7860")
        print("   Docs: http://localhost:7860/docs")
        print("   Features: REST API endpoints")
        run_fastapi()

if __name__ == "__main__":
    main()