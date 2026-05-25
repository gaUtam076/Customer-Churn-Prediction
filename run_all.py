#!/usr/bin/env python3
# ============================================================
# FILE: run_all.py
# DESCRIPTION:
# Master pipeline script for Customer Churn Prediction Project
#
# This script automatically runs:
#   1. Data Cleaning & EDA
#   2. Machine Learning Model Training
#   3. Power BI Data Export
#
# AUTHOR: Gautam
# ============================================================

import subprocess
import sys
import os


# ============================================================
# FUNCTION: RUN PYTHON SCRIPT
# ============================================================

def run_script(script_path, step_name):

    print("\n" + "=" * 65)
    print(f" {step_name}")
    print("=" * 65)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            check=True
        )

        print(f"\n✅ SUCCESS: {step_name} completed successfully.")

    except subprocess.CalledProcessError:
        print(f"\n ERROR while running: {script_path}")
        print("Please check the script and fix the issue.")
        sys.exit(1)


# ============================================================
# PROJECT START
# ============================================================

print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║        CUSTOMER CHURN PREDICTION PROJECT PIPELINE         ║
║                                                            ║
║        Running complete analytics workflow...             ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")


# ============================================================
# STEP 1 — DATA CLEANING & EDA
# ============================================================

run_script(
    os.path.join("notebooks", "01_data_cleaning_eda.py"),
    "STEP 1/3 → Data Cleaning & Exploratory Data Analysis"
)


# ============================================================
# STEP 2 — MACHINE LEARNING MODEL TRAINING
# ============================================================

run_script(
    os.path.join("notebooks", "02_model_training.py"),
    "STEP 2/3 → Machine Learning Model Training & Evaluation"
)


# ============================================================
# STEP 3 — POWER BI DATA EXPORT
# ============================================================

run_script(
    os.path.join("dashboard", "export_for_powerbi.py"),
    "STEP 3/3 → Power BI Dashboard Data Export"
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print("""
╔════════════════════════════════════════════════════════════╗
║                                                            ║
║              PROJECT EXECUTION COMPLETED                  ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║  Dashboard dataset exported successfully                  ║
║  Machine learning model trained successfully              ║
║  EDA visualizations generated successfully                ║
║                                                            ║
║  OUTPUT DIRECTORIES:                                      ║
║                                                            ║
║  • Screenshots  → screenshots/                            ║
║  • ML Models    → models/                                 ║
║  • Power BI CSV → dashboard/churn_data_for_powerbi.csv    ║
║                                                            ║
║  NEXT STEP:                                               ║
║                                                            ║
║  Open Power BI Desktop and load the exported CSV file     ║
║  to build the interactive churn analytics dashboard.      ║
║                                                            ║
╚════════════════════════════════════════════════════════════╝
""")