"""
Run all analysis steps in order.
"""

import subprocess
import sys


scripts = [
    "01_eda.py",
    "02_preprocessing.py",
    "03_clustering.py",
    "04_classification.py",
    "05_threshold.py",
    "06_shap.py",
    "07_visualization.py",
]

for script in scripts:
    print(f"\nRunning {script}")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"Stopped because {script} failed.")
        sys.exit(result.returncode)

print("\nDone. Results are saved in the outputs folder.")
