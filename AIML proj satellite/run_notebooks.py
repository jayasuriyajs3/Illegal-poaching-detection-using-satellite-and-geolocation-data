"""
Script to run all Jupyter notebooks programmatically.
"""
import subprocess
import sys
import os

notebooks = [
    'main.ipynb',
    'data_generator.ipynb',
    'anomaly_detector.ipynb',
    'image_detector.ipynb',
    'fusion_engine.ipynb',
    'visualizer.ipynb',
    'evaluator.ipynb',
    'utils.ipynb'
]

def run_notebook(notebook_path):
    """Execute a notebook and return success status."""
    print(f"Running {notebook_path}...")
    
    # Try different ways to run jupyter
    commands = [
        ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path],
        ['python', '-m', 'jupyter', 'nbconvert', '--to', 'notebook', '--execute', notebook_path],
        ['python', '-m', 'nbconvert', '--to', 'notebook', '--execute', notebook_path]
    ]
    
    for cmd in commands:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )
            print(f"✅ {notebook_path} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            if cmd == commands[-1]:  # Last attempt
                print(f"❌ Error running {notebook_path}: {e.stderr}")
                return False
            continue  # Try next command
        except FileNotFoundError:
            if cmd == commands[-1]:  # Last attempt
                print("❌ Jupyter not found. Please install: pip install jupyter nbconvert")
                return False
            continue  # Try next command
    
    return False

if __name__ == "__main__":
    print("="*60)
    print("Running Jupyter Notebooks")
    print("="*60)
    
    # Run main notebook (which runs the full pipeline)
    if os.path.exists('main.ipynb'):
        success = run_notebook('main.ipynb')
        if success:
            print("\n✅ Pipeline completed! Check the output/ directory for results.")
        else:
            print("\n❌ Pipeline failed. Check the error messages above.")
    else:
        print("❌ main.ipynb not found!")
        sys.exit(1)

