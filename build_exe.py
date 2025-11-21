import os
import subprocess
import sys
import shutil
from pathlib import Path

def build_exe():
    """
    Builds the Finance Predictor application into a standalone executable.
    """
    print("🚀 Starting Build Process...")
    
    # Clean previous builds
    for d in ["build", "dist"]:
        if os.path.exists(d):
            shutil.rmtree(d)

    # PyInstaller Command
    # We need to collect streamlit's internal files
    import streamlit.web.cli as stcli
    st_path = os.path.dirname(stcli.__file__)
    
    cmd = [
        "pyinstaller",
        "--noconfirm",
        "--onefile",
        "--windowed",
        "--name=FinancePredictor",
        "--clean",
        # Streamlit hooks
        f"--add-data={st_path};streamlit",
        # Copy src code
        "--add-data=src;src",
        # Hidden imports often missed by PyInstaller
        "--hidden-import=streamlit",
        "--hidden-import=pandas",
        "--hidden-import=numpy",
        "--hidden-import=sklearn.utils._cython_blas",
        "--hidden-import=sklearn.neighbors.typedefs",
        "--hidden-import=sklearn.neighbors.quad_tree",
        "--hidden-import=sklearn.tree._utils",
        "--hidden-import=prophet",
        "--hidden-import=xgboost",
        "--hidden-import=gpt4all",
        # Main entry point
        "app.py"
    ]
    
    print(f"Executing: {' '.join(cmd)}")
    
    try:
        subprocess.check_call(cmd)
        print("\n✅ Build Successful! Check the 'dist' folder.")
        print("⚠️  NOTE: You must copy the 'models' folder to the same directory as the executable for the LLM to work.")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Build Failed: {e}")

if __name__ == "__main__":
    # Check if pyinstaller is installed
    try:
        import PyInstaller
    except ImportError:
        print("Installing PyInstaller...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        
    build_exe()
