import torch
import sys
import os
import glob
import json

from gap_l2_normalization import apply_feature_regularization
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import FEATURE_LOGIT_OUTPUT, FEATURE_OUTPUT

def extract_confident_features(features: torch.Tensor, logits: torch.Tensor, top_k: int = 300) -> torch.Tensor:
    """Filters out background queries and returns the top-k most confident features."""
    features = features.squeeze(0) if features.dim() == 3 else features
    logits = logits.squeeze(0) if logits.dim() == 3 else logits

    confidences, _ = torch.max(logits, dim=-1)
    _, topk_indices = torch.topk(confidences, k=top_k, largest=True)
    
    return features[topk_indices]

def fit_mahalanobis_parameters(features: torch.Tensor) -> None:
    """Compute and store the mean and inverse covariance from training features"""
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / (features.shape[0] - 1)

    # Small regularisation to prevent singular covariance for low-rank inputs
    cov = cov + 1e-6 * torch.eye(features.shape[1], dtype=features.dtype)
    inv_cov = torch.linalg.inv(cov)
    return mean, inv_cov

def compute_mahalanobis_distance(features: torch.Tensor, mean: torch.Tensor, inv_cov: torch.Tensor) -> torch.Tensor:
    """Compute Mahalanobis distance for each feature vector."""
    delta = features - mean
    scores = torch.sqrt((delta @ inv_cov * delta).sum(dim=1))
    return scores.mean().item()

def save_mahalanobis_scores(results: dict, save_dir: str):
    """Saves the calculated mahalanobis distance to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "mahalanobis_distances.json")
    
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {len(results)} mahalanobis distances to: {file_path}")


if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')
    normalization = os.environ.get('NORMALIZATION', 'False').lower() in ('true', '1', 't')

    # ==========================================
    # DYNAMIC ROUTING & TIMESTAMP LOGIC
    # ==========================================
    if not all([weather, severity, timestamp]):
        print("\n[!] CRITICAL ERROR: Missing Environment Variables.")
        print("Please run the script like this:")
        print("OOD_WEATHER=Fog OOD_SEVERITY=easy OOD_TIMESTAMP=20260501_000617 python safety_monitor/energy_score.py\n")
        sys.exit(1)

    if weather == "Clear" and severity == "baseline":
        target_dir = os.path.join(FEATURE_LOGIT_OUTPUT, "nuscenes", timestamp)

        if normalization:
            feature_save_location = os.path.join(FEATURE_OUTPUT, "nuscenes", "normalized", timestamp)
        else:
            feature_save_location = os.path.join(FEATURE_OUTPUT, "nuscenes", timestamp)
    
        baseline_timestamp = timestamp

    else:
        target_dir = os.path.join(FEATURE_LOGIT_OUTPUT, weather, severity, timestamp)

        if normalization:
            feature_save_location = os.path.join(FEATURE_OUTPUT, weather, severity, "normalized", timestamp)
        else:
            feature_save_location = os.path.join(FEATURE_OUTPUT, weather, severity, timestamp)
        
        baseline_timestamp = os.environ.get('BASELINE_TIMESTAMP')

        if not baseline_timestamp:
            print("\n[!] CRITICAL ERROR: 'BASELINE_TIMESTAMP' is missing.")
            print("When evaluating corrupted data, you must provide the timestamp of your clean baseline.")
            sys.exit(1)

    if not os.path.exists(target_dir):
        print("ERROR: Directory does not exist! Check your spelling or timestamps.")
        sys.exit(1)

    pt_files = glob.glob(os.path.join(target_dir, "*.pt"))
    if not pt_files:
        print("No .pt files found in the target directory.")
        sys.exit(1)

    if normalization:
        baseline_param_file = os.path.join(FEATURE_OUTPUT, "nuscenes", "normalized", baseline_timestamp, "mahalanobis_baseline.pt")
    else:
        baseline_param_file = os.path.join(FEATURE_OUTPUT, "nuscenes", baseline_timestamp, "mahalanobis_baseline.pt")

    mean = None
    inv_cov = None

    # ==========================================
    # PHASE 1: FIT THE BASELINE (Clear Data)
    # ==========================================
    if weather == "Clear" and severity == "baseline":

        # Skip fitting if it has been already done (mahalanobis_baseline file already exists)
        if os.path.exists(baseline_param_file):
            print(f"Baseline already exists at {baseline_param_file}. Skipping fitting phase...")
            baseline_data = torch.load(baseline_param_file, map_location='cpu')
            mean = baseline_data['mean']
            inv_cov = baseline_data['inv_cov']
        else:
            print(f"Fitting Mahalanobis Baseline on {len(pt_files)} frames...")
            all_features_list = []
        
            for file_path in pt_files:
                try:
                    loaded_data = torch.load(file_path, map_location='cpu')
                    feats = extract_confident_features(loaded_data['features'], loaded_data['logits'])

                    if normalization:
                        feats = apply_feature_regularization(feats, use_gap=True, use_l2=True)

                    all_features_list.append(feats)
                except Exception as e:
                    print(f"Failed to read {file_path} for fitting: {e}")
                
            all_features_tensor = torch.cat(all_features_list, dim=0)
            mean, inv_cov = fit_mahalanobis_parameters(all_features_tensor)
        
            os.makedirs(os.path.dirname(baseline_param_file), exist_ok=True)
            torch.save({'mean': mean, 'inv_cov': inv_cov}, baseline_param_file)
            print(f"Baseline parameters saved to: {baseline_param_file}\n")

    # ==========================================
    # PHASE 2: LOAD THE BASELINE (Corrupted Data)
    # ==========================================
    else:
        print("Loading Baseline Parameters...")
        if not os.path.exists(baseline_param_file):
            print(f"[!] ERROR: Missing baseline file at {baseline_param_file}. Run on 'Clear/baseline' first!")
            sys.exit(1)
            
        baseline_data = torch.load(baseline_param_file, map_location='cpu')
        mean = baseline_data['mean']
        inv_cov = baseline_data['inv_cov']

    # ==========================================
    # SCORE ALL FRAMES (Runs for both modes)
    # ==========================================
    frame_results = {}
    print(f"Scoring {len(pt_files)} frames...")

    for file_path in pt_files:
        try:
            loaded_data = torch.load(file_path, map_location='cpu')

            sample_token = loaded_data['sample_token']
            features_tensor = loaded_data['features']
            logits_tensor = loaded_data['logits']

            valid_features = extract_confident_features(features_tensor, logits_tensor)

            if normalization:
                valid_features = apply_feature_regularization(valid_features, use_gap=True, use_l2=True)
            score = compute_mahalanobis_distance(valid_features, mean, inv_cov)

            frame_results[sample_token] = score

            
        except Exception as e:
            print(f"Failed to process {os.path.basename(file_path)}: {e}")

    print("Processing Complete!")
    save_mahalanobis_scores(frame_results, feature_save_location)

    print(f"Successfully calculated distances for {len(frame_results)} frames.")
    
    print("\n--- Sample Results ---")
    for idx, (token, score) in enumerate(frame_results.items()):
        if idx >= 5:
            break
        print(f"Token: {token} | Distance: {score:.4f}")

