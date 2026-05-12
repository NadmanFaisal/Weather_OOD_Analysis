# src: https://www.geeksforgeeks.org/machine-learning/false-positive-rate/

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_OUTPUT, ENERGY_OUTPUT, FPR95_PLOT_OUTPUT

def get_fpr95(fpr, tpr):
    idx = np.argmax(tpr >= 0.95)
    return fpr[idx]

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
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        return fpr, tpr
    except Exception as e:
        print(f"[!] Error calculating AUROC for {metric_name}: {e}")
        return None, None

# Generated code
def plot_fpr95_bar_chart(results: dict, weather: str, severity: str, save_dir: str, is_normalized: bool):
    plt.figure(figsize=(6, 6))
    
    metrics = list(results.keys())
    fpr95_values = [results[m] for m in metrics]
    color_map = {
        'Mahalanobis (Raw)': 'darkorange', 
        'Mahalanobis (Normalized)': 'firebrick', 
        'Energy Score': 'cornflowerblue'
    }

    bar_colors = [color_map.get(m, 'gray') for m in metrics]

    bars = plt.bar(metrics, fpr95_values, color=bar_colors, width=0.5)

    plt.ylim([0.0, 1.05])
    plt.ylabel('FPR95 (Lower is Better)')

    title_suffix = " (Normalized)" if is_normalized else " (Raw)"
    plt.title(f'FPR95 Evaluation: {weather.capitalize()} ({severity.capitalize()}){title_suffix}')
    
    plt.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='Ideal FPR (5%)')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)

    plt.bar_label(bars, fmt='%.4f', padding=3)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"fpr95_bar_{weather}_{severity}.png")
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"[*] FPR95 Bar Chart saved successfully to: {file_path}")
    plt.close()

if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')
    baseline_timestamp = os.environ.get('BASELINE_TIMESTAMP')
    normalization = os.environ.get('NORMALIZATION', 'False').lower() in ('true', '1', 't')

    id_energy_path = os.path.join(ENERGY_OUTPUT, "nuscenes", baseline_timestamp, "energy_scores.json")
    ood_energy_path = os.path.join(ENERGY_OUTPUT, weather, severity, timestamp, "energy_scores.json")

    results = {}

    id_maha_raw_path = os.path.join(FEATURE_OUTPUT, "nuscenes", baseline_timestamp, "mahalanobis_distances.json")
    ood_maha_raw_path = os.path.join(FEATURE_OUTPUT, weather, severity, timestamp, "mahalanobis_distances.json")

    # Evaluate Mahalanobis Distance
    if os.path.exists(id_maha_raw_path) and os.path.exists(ood_maha_raw_path):
        id_maha_raw_scores = get_json_scores(id_maha_raw_path)
        ood_maha_raw_scores = get_json_scores(ood_maha_raw_path)
        
        maha_raw_fpr, maha_raw_tpr = get_auroc_score(id_maha_raw_scores, ood_maha_raw_scores, "Mahalanobis (Raw)")
        if maha_raw_fpr is not None:
            maha_raw_fpr95 = get_fpr95(maha_raw_fpr, maha_raw_tpr)
            results['Mahalanobis (Raw)'] = maha_raw_fpr95
            print(f"Mahalanobis Distances Loaded -> ID: {len(id_maha_raw_scores)} frames | OOD: {len(ood_maha_raw_scores)} frames")
    else:
        print("[!] Missing Mahalanobis JSON files. Skipping metric.")
        if not os.path.exists(id_maha_raw_path): 
            print(f"\tMissing ID: {id_maha_raw_path}")
        if not os.path.exists(ood_maha_raw_path): 
            print(f"\tMissing OOD: {ood_maha_raw_path}")

    if normalization:
        # Evaluate Mahalanobis (Normalized)
        id_maha_norm_path = os.path.join(FEATURE_OUTPUT, "nuscenes", "normalized", baseline_timestamp, "mahalanobis_distances.json")
        ood_maha_norm_path = os.path.join(FEATURE_OUTPUT, weather, severity, "normalized", timestamp, "mahalanobis_distances.json")

        if os.path.exists(id_maha_norm_path) and os.path.exists(ood_maha_norm_path):
            id_maha_norm_scores = get_json_scores(id_maha_norm_path)
            ood_maha_norm_scores = get_json_scores(ood_maha_norm_path)
            
            maha_norm_fpr, maha_norm_tpr = get_auroc_score(id_maha_norm_scores, ood_maha_norm_scores, "Mahalanobis (Normalized)")
            if maha_norm_fpr is not None:
                maha_norm_fpr95 = get_fpr95(maha_norm_fpr, maha_norm_tpr)
                results['Mahalanobis (Normalized)'] = maha_norm_fpr95
                print(f"Normalized Mahalanobis Loaded -> ID: {len(id_maha_norm_scores)} frames | OOD: {len(ood_maha_norm_scores)} frames")
        else:
            print("[!] Missing Mahalanobis JSON files. Skipping metric.")
            if not os.path.exists(id_maha_norm_path): 
                print(f"\tMissing ID: {id_maha_norm_path}")
            if not os.path.exists(ood_maha_norm_path): 
                print(f"\tMissing OOD: {ood_maha_norm_path}")

    else:
        # Evaluate Energy Score
        if os.path.exists(id_energy_path) and os.path.exists(ood_energy_path):
            id_energy_scores = get_json_scores(id_energy_path)
            ood_energy_scores = get_json_scores(ood_energy_path)
            
            energy_fpr, energy_tpr = get_auroc_score(id_energy_scores, ood_energy_scores, "Energy Score")
            if energy_fpr is not None:
                energy_fpr95 = get_fpr95(energy_fpr, energy_tpr)

                results['Energy Score'] = energy_fpr95

                print(f"Energy Scores Loaded -> ID: {len(id_energy_scores)} frames | OOD: {len(ood_energy_scores)} frames")
        else:
            print("[!] Missing Energy JSON files. Skipping metric.")
            if not os.path.exists(id_energy_path): 
                print(f"\tMissing ID: {id_energy_path}")
            if not os.path.exists(ood_energy_path): 
                print(f"\tMissing OOD: {ood_energy_path}")

    if results:
        if normalization:
            save_dir = os.path.join(FPR95_PLOT_OUTPUT, weather, severity, "normalized", timestamp)
        else:
            save_dir = os.path.join(FPR95_PLOT_OUTPUT, weather, severity, timestamp)

        plot_fpr95_bar_chart(results, weather, severity, save_dir, normalization)

    # Print the final benchmarks
    print("\n---------------- FINAL FPR95 ---------------------")
    if not results:
        print("No valid scores were found to evaluate. Please check the paths above.")
    else:
        for metric, scores in results.items():
            print(f"{metric} | FPR95: {scores}")
