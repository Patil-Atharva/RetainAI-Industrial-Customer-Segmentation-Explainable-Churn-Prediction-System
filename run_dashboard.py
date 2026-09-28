"""
Launcher script for RetainAI Streamlit Analytics Web Dashboard.
Executes: streamlit run src/dashboard/app.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    dashboard_app = Path(__file__).resolve().parent / "src" / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(dashboard_app)]
    subprocess.run(cmd)


if __name__ == "__main__":
    main()
