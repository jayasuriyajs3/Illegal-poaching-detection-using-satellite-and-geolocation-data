"""
Launch Dashboard Script
Opens the project dashboard in the default web browser
"""

import webbrowser
import os
from pathlib import Path

def launch_dashboard():
    """Launch the dashboard in the default web browser"""
    
    # Get the dashboard file path
    dashboard_path = Path(__file__).parent / "dashboard.html"
    
    if not dashboard_path.exists():
        print("❌ Error: dashboard.html not found!")
        print(f"Expected location: {dashboard_path}")
        return
    
    # Convert to absolute path and file URL
    abs_path = dashboard_path.resolve()
    file_url = f"file:///{abs_path}".replace("\\", "/")
    
    print("=" * 60)
    print("🛰️  ILLEGAL POACHING DETECTION SYSTEM - DASHBOARD")
    print("=" * 60)
    print()
    print("📊 Opening dashboard in your default web browser...")
    print(f"📁 Dashboard location: {abs_path}")
    print()
    print("✨ Features available:")
    print("   • Real-time statistics and metrics")
    print("   • Interactive charts and visualizations")
    print("   • Recent alerts table")
    print("   • Technology stack overview")
    print("   • System performance information")
    print("   • Quick access to interactive map")
    print()
    
    try:
        # Open the dashboard in default browser
        webbrowser.open(file_url)
        print("✅ Dashboard launched successfully!")
        print()
        print("💡 Tip: Bookmark this page for quick access")
        print("🔄 Click 'Refresh Data' button to reload statistics")
        print("🗺️  Click 'Open Interactive Map' to view geospatial data")
        print()
        
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        print()
        print("Alternative: Open this file manually in your browser:")
        print(f"   {abs_path}")

if __name__ == "__main__":
    launch_dashboard()
