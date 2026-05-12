# src: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_OUTPUT, ENERGY_OUTPUT, AUROC_PLOT_OUTPUT

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
        fpr, tpr, thresholds = roc_curve(y_true, y_score)

        return auroc, fpr, tpr
    except Exception as e:
        print(f"[!] Error calculating AUROC for {metric_name}: {e}")
        return None, None, None

# Generated code
def plot_roc_curves(curve_data: dict, weather: str, severity: str, save_dir, is_normalized: bool):
    plt.figure(figsize=(8, 6))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (0.50)')

    colors = {
        'Mahalanobis Distance': 'darkorange', 
        'Mahalanobis (Normalized)': 'firebrick', 
        'Energy Score': 'cornflowerblue'
    }
    
    for metric_name, data in curve_data.items():
        fpr = data['fpr']
        tpr = data['tpr']
        auroc = data['auroc']
        color = colors.get(metric_name, 'green')
        
        plt.plot(fpr, tpr, color=color, lw=2, 
                 label=f'{metric_name} (AUC = {auroc:.4f})')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')

    title_suffix = " (Raw vs Normalized)" if is_normalized else " (Raw vs Energy)"
    plt.title(f'OOD Detection ROC Curve: {weather.capitalize()} ({severity.capitalize()}){title_suffix}')

    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)

    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, f"roc_curve_{weather}_{severity}.png")
    
    plt.savefig(file_path, dpi=300, bbox_inches='tight')
    print(f"ROC Curve saved successfully to: {file_path}")
    plt.close()

if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')
    baseline_timestamp = os.environ.get('BASELINE_TIMESTAMP')
    normalization = os.environ.get('NORMALIZATION', 'False').lower() in ('true', '1', 't')

    if not all([weather, severity, timestamp, baseline_timestamp]):
        print("\n[!] CRITICAL ERROR: Missing Environment Variables.")
        print("Please run the script like this:")
        print("BASELINE_TIMESTAMP=20260502_220411 OOD_WEATHER=Fog OOD_SEVERITY=hard OOD_TIMESTAMP=... python safety_monitor/auroc_evaluator.py\n")
        sys.exit(1)

    if normalization:
        id_maha_path = os.path.join(FEATURE_OUTPUT, "nuscenes", "normalized", baseline_timestamp, "mahalanobis_distances.json")
        ood_maha_path = os.path.join(FEATURE_OUTPUT, weather, severity, "normalized", timestamp, "mahalanobis_distances.json")
    else:
        id_maha_path = os.path.join(FEATURE_OUTPUT, "nuscenes", baseline_timestamp, "mahalanobis_distances.json")
        ood_maha_path = os.path.join(FEATURE_OUTPUT, weather, severity, timestamp, "mahalanobis_distances.json")

    id_maha_raw_path = os.path.join(FEATURE_OUTPUT, "nuscenes", baseline_timestamp, "mahalanobis_distances.json")
    ood_maha_raw_path = os.path.join(FEATURE_OUTPUT, weather, severity, timestamp, "mahalanobis_distances.json")

    results = {}
    curve_data = {}

    # Evaluate Mahalanobis Distance
    if os.path.exists(id_maha_path) and os.path.exists(ood_maha_path):
        id_maha_raw_scores = get_json_scores(id_maha_raw_path)
        ood_maha_raw_scores = get_json_scores(ood_maha_raw_path)
        
        maha_raw_auroc, maha_raw_fpr, maha_raw_tpr = get_auroc_score(id_maha_raw_scores, ood_maha_raw_scores, "Mahalanobis (Raw)")
        if maha_raw_auroc is not None:
            results['Mahalanobis (Raw)'] = maha_raw_auroc
            curve_data['Mahalanobis (Raw)'] = {'fpr': maha_raw_fpr, 'tpr': maha_raw_tpr, 'auroc': maha_raw_auroc}
            print(f"Raw Mahalanobis Loaded -> ID: {len(id_maha_raw_scores)} frames | OOD: {len(ood_maha_raw_scores)} frames")
    else:
        print("[!] Missing Mahalanobis JSON files. Skipping metric.")
        if not os.path.exists(id_maha_path): 
            print(f"\tMissing ID: {id_maha_path}")
        if not os.path.exists(ood_maha_path): 
            print(f"\tMissing OOD: {ood_maha_path}")

    if normalization:
        # Evaluate Mahalanobis raw vs Mahalanobis normalized
        id_maha_norm_path = os.path.join(FEATURE_OUTPUT, "nuscenes", "normalized", baseline_timestamp, "mahalanobis_distances.json")
        ood_maha_norm_path = os.path.join(FEATURE_OUTPUT, weather, severity, "normalized", timestamp, "mahalanobis_distances.json")

        if os.path.exists(id_maha_norm_path) and os.path.exists(ood_maha_norm_path):
            id_maha_norm_scores = get_json_scores(id_maha_norm_path)
            ood_maha_norm_scores = get_json_scores(ood_maha_norm_path)
            
            maha_norm_auroc, maha_norm_fpr, maha_norm_tpr = get_auroc_score(id_maha_norm_scores, ood_maha_norm_scores, "Mahalanobis (Normalized)")
            if maha_norm_auroc is not None:
                results['Mahalanobis (Normalized)'] = maha_norm_auroc
                curve_data['Mahalanobis (Normalized)'] = {'fpr': maha_norm_fpr, 'tpr': maha_norm_tpr, 'auroc': maha_norm_auroc}
                print(f"Normalized Mahalanobis Loaded -> ID: {len(id_maha_norm_scores)} frames | OOD: {len(ood_maha_norm_scores)} frames")
        else:
            print("[!] Missing Normalized Mahalanobis JSON files.")
    else:
        # Evaluate Mahalanobis raw vs Energy Score
        id_energy_path = os.path.join(ENERGY_OUTPUT, "nuscenes", baseline_timestamp, "energy_scores.json")
        ood_energy_path = os.path.join(ENERGY_OUTPUT, weather, severity, timestamp, "energy_scores.json")

        if os.path.exists(id_energy_path) and os.path.exists(ood_energy_path):
            id_energy_scores = get_json_scores(id_energy_path)
            ood_energy_scores = get_json_scores(ood_energy_path)
            
            energy_auroc, energy_fpr, energy_tpr = get_auroc_score(id_energy_scores, ood_energy_scores, "Energy Score")
            if energy_auroc is not None:
                results['Energy Score'] = energy_auroc

                curve_data['Energy Score'] = {'fpr': energy_fpr, 'tpr': energy_tpr, 'auroc': energy_auroc}
                print(f"Energy Scores Loaded -> ID: {len(id_energy_scores)} frames | OOD: {len(ood_energy_scores)} frames")
        else:
            print("[!] Missing Energy JSON files. Skipping metric.")
            if not os.path.exists(id_energy_path): 
                print(f"\tMissing ID: {id_energy_path}")
            if not os.path.exists(ood_energy_path): 
                print(f"\tMissing OOD: {ood_energy_path}")

    if curve_data:
        if normalization:
            save_dir = os.path.join(AUROC_PLOT_OUTPUT, weather, severity, "normalized", timestamp)
        else:
            save_dir = os.path.join(AUROC_PLOT_OUTPUT, weather, severity, timestamp)
        
        plot_roc_curves(curve_data, weather, severity, save_dir, normalization)

    # Print the final benchmarks
    print("\n---------------- FINAL AUROC ---------------------")
    if not results:
        print("No valid scores were found to evaluate. Please check the paths above.")
    else:
        for metric, auroc in results.items():
            print(f"{metric}: {auroc}")
