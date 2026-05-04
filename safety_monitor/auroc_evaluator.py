import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_OUTPUT, ENERGY_OUTPUT

def get_json_scores(file):
    with open(file) as json_data:
        data = json.load(json_data)
        json_data.close()

    return list(data.values())

# def get_auroc_score():

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

    print("ID Mahalanobis Distances:", id_maha_scores)
    print("ID Energy Scores:", id_energy_scores)
