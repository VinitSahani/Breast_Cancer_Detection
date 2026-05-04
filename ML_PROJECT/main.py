"""
=============================================================================
MAIN ORCHESTRATOR — Runs all agents sequentially and saves artifacts
=============================================================================
Usage: python main.py
=============================================================================
"""

import os
import sys
import joblib

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import load_and_preprocess
from notebooks.eda_analysis import run_eda
from models.train import train_and_evaluate


def main():
    print("\n" + "=" * 65)
    print("  BREAST CANCER DETECTION - ML PIPELINE")
    print("=" * 65)

    # ── Agent 1: Data & Preprocessing ──
    agent1_output = load_and_preprocess()

    # ── Agent 2: EDA & Feature Analysis ──
    agent2_output = run_eda(agent1_output)

    # ── Agent 3: Model Training & Evaluation ──
    agent3_output = train_and_evaluate(agent1_output)

    # ── Save artifacts for deployment (Agent 4) ──
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    model_path  = os.path.join(models_dir, "best_model.pkl")
    scaler_path = os.path.join(models_dir, "scaler.pkl")

    joblib.dump(agent3_output["best_model"], model_path)
    joblib.dump(agent1_output["scaler"], scaler_path)

    print("\n" + "=" * 65)
    print("  💾  SAVED ARTIFACTS")
    print("=" * 65)
    print(f"   Model  : {model_path}")
    print(f"   Scaler : {scaler_path}")

    # ── Final summary ──
    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("=" * 65)
    print(f"   Best Model : {agent3_output['best_model_name']}")
    print(f"   Recall     : {agent3_output['results'][agent3_output['best_model_name']]['recall']}")
    print(f"   F1-Score   : {agent3_output['results'][agent3_output['best_model_name']]['f1_score']}")
    print(f"   Plots      : notebooks/plots/")
    print(f"   Streamlit  : streamlit run app.py")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
