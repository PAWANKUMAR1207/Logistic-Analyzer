#!/usr/bin/env python3
"""
Local deployment script for Logistics Deliveries Analytics Dashboard
Run this script to start the dashboard on port 7000
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    packages = [
        'streamlit>=1.28.0',
        'pandas>=2.0.0', 
        'plotly>=5.15.0',
        'numpy>=1.24.0'
    ]
    
    for package in packages:
        try:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
            return False
    return True

def run_dashboard():
    """Run the Streamlit dashboard on port 7000"""
    try:
        print("Starting Logistics Analytics Dashboard on port 7000...")
        print("Access the dashboard at: http://localhost:7000")
        print("Press Ctrl+C to stop the server")
        subprocess.run([
            'python', '-m', 'streamlit', 'run', 'app.py', 
            '--server.port', '7000',
            '--server.address', '0.0.0.0'
        ])
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    except subprocess.CalledProcessError as e:
        print(f"Error running dashboard: {e}")

if __name__ == "__main__":
    print("=== Logistics Deliveries Analytics Dashboard ===")
    print("Setting up local environment...")
    
    if install_requirements():
        print("All dependencies installed successfully!")
        run_dashboard()
    else:
        print("Failed to install dependencies. Please install manually:")
        print("pip install streamlit pandas plotly numpy")