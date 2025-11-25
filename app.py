"""
ForenSmart Main Application
===========================

Main entry point that runs the dashboard with all features:
- Consent management
- Data extraction
- Intelligence analysis
- Report generation
- Storage management
- Diagnostics

Run with: streamlit run app.py
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import and run the dashboard
from modules.dashboard_merged import main

if __name__ == "__main__":
    main()
