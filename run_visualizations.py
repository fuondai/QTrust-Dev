#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization Runner for QTrust Project

This script automates the process of generating and organizing visualizations for the QTrust blockchain framework.
It runs multiple visualization scripts, ensures output directories exist, and generates an HTML gallery
to display all created charts in a user-friendly format.
"""

import os
import subprocess
import sys
from pathlib import Path
import glob
import time

def ensure_directories():
    """Create necessary directories for visualization outputs."""
    directories = [
        "docs/exported_charts",
        "docs/architecture",
        "data/sources",
        "data/visualization"
    ]
    
    print("Ensuring required directories exist...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  - {directory}")
    print()

def run_script(script_path):
    """Execute a Python script and print the execution status."""
    print(f"========== Generating charts from {os.path.basename(script_path)} ==========")
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print(f"Success! Exit code: {result.returncode}")
        print(result.stdout)
    else:
        print(f"Error! Exit code: {result.returncode}")
        print(result.stderr)
    
    print()
    return result.returncode == 0

def create_image_gallery():
    """Create an HTML gallery to display all charts."""
    charts_dir = "docs/exported_charts"
    output_file = os.path.join(charts_dir, "index.html")
    
    # Get all PNG files in the charts directory
    image_files = glob.glob(os.path.join(charts_dir, "*.png"))
    
    # HTML content
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QTrust Visualization Gallery</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            padding: 20px;
            background-color: #f8f9fa;
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-container {
            margin-bottom: 30px;
            background-color: #fff;
            border-radius: 5px;
            padding: 15px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .chart-img {
            width: 100%;
            height: auto;
            border: 1px solid #eee;
        }
        .chart-title {
            margin-top: 10px;
            font-weight: bold;
        }
        .footer {
            text-align: center;
            margin-top: 30px;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .reference-section {
            margin: 30px 0;
            padding: 20px;
            background-color: #fff;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>QTrust Visualization Gallery</h1>
            <p class="lead">Performance metrics and architecture diagrams for the QTrust blockchain framework</p>
        </div>
        
        <div class="row">
"""
    
    # Add each image to the gallery
    for i, img_path in enumerate(sorted(image_files)):
        img_name = os.path.basename(img_path)
        # Format the title from filename (remove extension and replace underscores)
        title = os.path.splitext(img_name)[0].replace('_', ' ').title()
        
        html_content += f"""
            <div class="col-md-6">
                <div class="chart-container">
                    <img src="{img_name}" class="chart-img" alt="{title}">
                    <div class="chart-title">{title}</div>
                </div>
            </div>"""
        
        # Add a clearfix after every 2 images
        if (i + 1) % 2 == 0:
            html_content += """
        </div>
        <div class="row">"""
    
    # Close the row if there are an odd number of images
    html_content += """
        </div>
        
        <div class="reference-section">
            <h3>Data Sources</h3>
            <ul>
                <li>QTrust Benchmark Results (2025)</li>
                <li>Performance comparison with Ethereum 2.0, Polkadot, Solana, and Algorand</li>
                <li>Internal testing and simulation results</li>
                <li>QTrust Security Team Analysis (2025)</li>
            </ul>
        </div>
        
        <div class="footer">
            <p>&copy; 2025 QTrust Blockchain Framework</p>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""
    
    # Write the HTML file
    with open(output_file, 'w') as f:
        f.write(html_content)
    
    print(f"HTML gallery created at: {output_file}")

def main():
    """Run all visualization scripts and create HTML gallery."""
    start_time = time.time()
    
    print("========== GENERATING CHARTS FOR QTRUST ==========")
    ensure_directories()
    
    # List of scripts to run
    scripts = [
        "data/visualization/generate_performance_charts.py",
        "data/visualization/generate_architecture_diagram.py",
        "data/visualization/generate_attack_resilience.py",
        "data/visualization/generate_federated_learning.py",
        "data/visualization/generate_caching_charts.py"
    ]
    
    success_count = 0
    failed_scripts = []
    
    # Run each script
    for script in scripts:
        success = run_script(script)
        if success:
            success_count += 1
        else:
            failed_scripts.append(script)
    
    # Create HTML gallery
    create_image_gallery()
    
    # Print summary
    print("========== CHART GENERATION RESULTS ==========")
    print(f"Total scripts: {len(scripts)}")
    print(f"Successfully completed: {success_count}")
    print(f"Failed: {len(failed_scripts)}")
    
    if failed_scripts:
        print("\nFailed scripts:")
        for script in failed_scripts:
            print(f"  - {script}")
    
    end_time = time.time()
    print(f"\nTotal time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 