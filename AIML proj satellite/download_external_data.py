"""
Quick script to download external dataset for project review.
This demonstrates external data source integration.
"""
import subprocess
import sys

print("="*70)
print("📥 DOWNLOADING EXTERNAL DATASET")
print("="*70)
print()
print("This will download a small wildlife dataset (~200 MB) from Kaggle")
print("to demonstrate external data source usage for project review.")
print()

response = input("Continue with download? (y/n): ")
if response.lower() != 'y':
    print("Download cancelled.")
    sys.exit(0)

print()
print("Executing download_dataset.ipynb...")
print()

# Run the notebook
cmd = [
    sys.executable, '-m', 'jupyter', 'nbconvert',
    '--to', 'notebook',
    '--execute', 'download_dataset.ipynb',
    '--output', 'download_dataset_executed.ipynb'
]

try:
    subprocess.run(cmd, check=True)
    print()
    print("="*70)
    print("✅ DATASET DOWNLOAD COMPLETE")
    print("="*70)
    print()
    print("Dataset location: external_dataset/")
    print("Documentation: DATASET_INFO.md")
    print()
    print("You can now run the main pipeline with:")
    print("   python run_project.py")
    print()
except Exception as e:
    print(f"Error: {e}")
    print()
    print("Alternative: Open download_dataset.ipynb in Jupyter/VS Code and run manually")
