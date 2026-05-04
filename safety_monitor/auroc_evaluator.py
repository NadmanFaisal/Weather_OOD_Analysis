# src: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

import os
import sys
import json
import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_OUTPUT, ENERGY_OUTPUT

def get_json_scores(file):
    with open(file) as json_data:
        data = json.load(json_data)
        json_data.close()

    return list(data.values())

def get_auroc_score(id_scores, ood_scores, metric_name):
    id_labels = np.zeros(len(id_scores))
    ood_labels = np.ones(len(ood_scores))

    y_true = np.concatenate([id_labels, ood_labels])
    y_score = np.concatenate([id_scores, ood_scores])

    try:
        auroc = roc_auc_score(y_true, y_score)
        return auroc
    except Exception as e:
        print(f"[!] Error calculating AUROC for {metric_name}: {e}")
        return None

if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')
    baseline_timestamp = os.environ.get('BASELINE_TIMESTAMP')

    if not all([weather, severity, timestamp, baseline_timestamp]):
        print("\n[!] CRITICAL ERROR: Missing Environment Variables.")
        print("Please run the script like this:")
        print("BASELINE_TIMESTAMP=20260502_220411 OOD_WEATHER=Fog OOD_SEVERITY=hard OOD_TIMESTAMP=... python safety_monitor/auroc_evaluator.py\n")
        sys.exit(1)

    id_maha_path = os.path.join(FEATURE_OUTPUT, "nuscenes", baseline_timestamp, "mahalanobis_distances.json")
    ood_maha_path = os.path.join(FEATURE_OUTPUT, weather, severity, timestamp, "mahalanobis_distances.json")

    id_energy_path = os.path.join(ENERGY_OUTPUT, "nuscenes", baseline_timestamp, "energy_scores.json")
    ood_energy_path = os.path.join(ENERGY_OUTPUT, weather, severity, timestamp, "energy_scores.json")

    id_maha_scores = get_json_scores(id_maha_path)
    ood_maha_scores = get_json_scores(ood_maha_path)

    id_energy_scores = get_json_scores(id_energy_path)
    ood_energy_scores = get_json_scores(ood_energy_path)

    results = {}

    # Evaluate Mahalanobis Distance
    if os.path.exists(id_maha_path) and os.path.exists(ood_maha_path):
        id_maha_scores = get_json_scores(id_maha_path)
        ood_maha_scores = get_json_scores(ood_maha_path)
        
        maha_auroc = get_auroc_score(id_maha_scores, ood_maha_scores, "Mahalanobis")
        if maha_auroc is not None:
            results['Mahalanobis Distance'] = maha_auroc
            print(f"Mahalanobis Distances Loaded -> ID: {len(id_maha_scores)} frames | OOD: {len(ood_maha_scores)} frames")
    else:
        print("[!] Missing Mahalanobis JSON files. Skipping metric.")
        if not os.path.exists(id_maha_path): 
            print(f"\tMissing ID: {id_maha_path}")
        if not os.path.exists(ood_maha_path): 
            print(f"\tMissing OOD: {ood_maha_path}")

    # Evaluate Energy Score
    if os.path.exists(id_energy_path) and os.path.exists(ood_energy_path):
        id_energy_scores = get_json_scores(id_energy_path)
        ood_energy_scores = get_json_scores(ood_energy_path)
        
        energy_auroc = get_auroc_score(id_energy_scores, ood_energy_scores, "Energy Score")
        if energy_auroc is not None:
            results['Energy Score'] = energy_auroc
            print(f"Energy Scores Loaded -> ID: {len(id_energy_scores)} frames | OOD: {len(ood_energy_scores)} frames")
    else:
        print("[!] Missing Energy JSON files. Skipping metric.")
        if not os.path.exists(id_energy_path): 
            print(f"\tMissing ID: {id_energy_path}")
        if not os.path.exists(ood_energy_path): 
            print(f"\tMissing OOD: {ood_energy_path}")

    # Print the final benchmarks
    print("\n---------------- FINAL AUROC ---------------------")
    if not results:
        print("No valid scores were found to evaluate. Please check the paths above.")
    else:
        for metric, auroc in results.items():
            print(f"{metric}: {auroc}")
