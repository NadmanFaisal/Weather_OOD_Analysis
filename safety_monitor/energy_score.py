import torch
import os
import sys
import glob
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import LOGIT_OUTPUT, ENERGY_OUTPUT

def compute_energy_score(logits: torch.Tensor, T: float = 1.0) -> torch.Tensor:
    """Compute the energy score for a batch of logits.

    Args:
        logits: Tensor of shape (N, K) — N samples, K classes.
        T: Temperature parameter (default 1.0).

    Returns:
        1-D tensor of shape (N,) containing the energy score per sample.
        Lower energy indicates in-distribution; higher indicates OOD.

    Note:
        Real logits will be injected via a PyTorch forward hook in a later issue.
        This implementation operates on dummy logits for development and testing.
    """
    query_scores = -T * torch.logsumexp(logits / T, dim=-1)
    energy_score = query_scores.min()
    return energy_score.detach()

def save_energy_scores(results: dict, save_dir: str):
    """Saves the calculated energy scores to a JSON file."""
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "energy_scores.json")
    
    with open(file_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"Saved {len(results)} scores to: {file_path}")

if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')

    if not all([weather, severity, timestamp]):
        print("\n[!] CRITICAL ERROR: Missing Environment Variables.")
        print("Please run the script like this:")
        print("OOD_WEATHER=Fog OOD_SEVERITY=easy OOD_TIMESTAMP=20260501_000617 python safety_monitor/energy_score.py\n")
        sys.exit(1)

    if weather == "Clear" and severity == "baseline":
        target_dir = os.path.join(LOGIT_OUTPUT, "nuscenes", timestamp)
        energy_save_location = os.path.join(ENERGY_OUTPUT, "nuscenes", timestamp)

    else:
        target_dir = os.path.join(LOGIT_OUTPUT, weather, severity, timestamp)
        energy_save_location = os.path.join(ENERGY_OUTPUT, weather, severity, timestamp)

    if not os.path.exists(target_dir):
        print("ERROR: Directory does not exist! Check your spelling or timestamps.")
        sys.exit(1)

    pt_files = glob.glob(os.path.join(target_dir, "*.pt"))
    if not pt_files:
        print("No .pt files found in the target directory.")
        sys.exit(1)

    frame_results = {}

    for file_path in pt_files:
        try:
            loaded_data = torch.load(file_path, map_location='cpu')
            
            sample_token = loaded_data['sample_token']
            logits_tensor = loaded_data['logits']
            
            score = compute_energy_score(logits_tensor).item()
            
            frame_results[sample_token] = score
            
        except Exception as e:
            print(f"Failed to process {os.path.basename(file_path)}: {e}")

    print("Processing Complete!")

    save_energy_scores(frame_results, energy_save_location)

    print(f"Successfully calculated energy scores for {len(frame_results)} frames.")
    
    print("\n--- Sample Results ---")
    for idx, (token, score) in enumerate(frame_results.items()):
        if idx >= 5:
            break
        print(f"Token: {token} | Energy: {score:.4f}")

    #model_state_dict = torch.load(file_path)
    #print("Model dict:", model_state_dict)
    #print("Shape of Model dict:", model_state_dict.shape)

    #energy_score = compute_energy_score(model_state_dict)
    #print("Energy score", energy_score)
    #print("Shape of energy score:", energy_score.shape)
