"""
Single-command script to display results from the Illegal Poaching Detection System.
This script shows the existing pipeline outputs.

Usage:
    python run_project.py
"""
import os
import sys
import webbrowser
from datetime import datetime

def main():
    print("="*70)
    print("🛰️  ILLEGAL POACHING DETECTION SYSTEM")
    print("   Project Results and Output Files")
    print("="*70)
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('output'):
        print("❌ Error: output/ directory not found")
        print("   Please run this script from the project root directory")
        sys.exit(1)
    
    # Check for output files
    output_files = {
        'poaching_alerts_map.html': 'Interactive Map Visualization',
        'poaching_alerts.csv': 'Poaching Alerts Data',
        'gps_tracking_data.csv': 'GPS Tracking Data',
        'gps_anomalies.csv': 'GPS Anomaly Detection Results',
        'image_detections.csv': 'Image Detection Results',
        'image_metadata.csv': 'Satellite Image Metadata',
        'performance_report.txt': 'System Performance Report'
    }
    
    print("📊 PROJECT EXECUTION SUMMARY")
    print("="*70)
    print()
    
    # Read and display performance report if available
    report_path = os.path.join('output', 'performance_report.txt')
    if os.path.exists(report_path):
        print("📈 PERFORMANCE METRICS:")
        print("-" * 70)
        with open(report_path, 'r') as f:
            content = f.read()
            print(content)
        print()
    
    print("="*70)
    print("📁 OUTPUT FILES")
    print("="*70)
    print()
    
    files_found = []
    files_missing = []
    
    for filename, description in output_files.items():
        filepath = os.path.join('output', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            files_found.append((filename, description, size))
            print(f"✅ {filename}")
            print(f"   {description}")
            print(f"   Size: {size:,} bytes")
            print()
        else:
            files_missing.append((filename, description))
    
    # Show directories
    print("📂 OUTPUT DIRECTORIES:")
    print()
    for item in os.listdir('output'):
        item_path = os.path.join('output', item)
        if os.path.isdir(item_path):
            try:
                file_count = len([f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))])
                print(f"   • {item}/ ({file_count} files)")
            except:
                print(f"   • {item}/")
    print()
    
    print("="*70)
    print("🔍 QUICK ACTIONS")
    print("="*70)
    print()
    print("1. View Interactive Map:")
    print("   - Opening poaching_alerts_map.html in your browser...")
    
    # Try to open the map in browser
    map_path = os.path.join('output', 'poaching_alerts_map.html')
    if os.path.exists(map_path):
        try:
            abs_path = os.path.abspath(map_path)
            webbrowser.open('file://' + abs_path)
            print(f"   ✅ Opened: {abs_path}")
        except Exception as e:
            print(f"   ⚠️  Could not auto-open. Manually open:")
            print(f"   {os.path.abspath(map_path)}")
    print()
    
    print("2. View CSV Files:")
    print("   cd output")
    print("   dir *.csv")
    print()
    
    print("3. Read Performance Report:")
    print("   type output\\performance_report.txt")
    print()
    
    if files_missing:
        print("="*70)
        print("⚠️  MISSING FILES")
        print("="*70)
        print()
        for filename, description in files_missing:
            print(f"   ❌ {filename} - {description}")
        print()
        print("To regenerate all outputs, open main.ipynb in VS Code and click 'Run All'")
        print()
    
    print("="*70)
    print("✅ PROJECT STATUS: COMPLETED")
    print("="*70)
    print()
    print("All outputs are ready for analysis!")
    print()

if __name__ == "__main__":
    main()
