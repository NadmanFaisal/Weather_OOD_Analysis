import torch
import os
import sys
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import LOGIT_OUTPUT


current_dir = os.path.dirname(os.path.abspath(__file__))

logit_dir_path = os.path.abspath(os.path.join(
    current_dir,
    "../",
    "data/intercepted_logits/"
))

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


if __name__ == "__main__":
    weather = os.environ.get('OOD_WEATHER')
    severity = os.environ.get('OOD_SEVERITY')
    timestamp = os.environ.get('OOD_TIMESTAMP')

    if not all([weather, severity, timestamp]):
        print("\n[!] CRITICAL ERROR: Missing Environment Variables.")
        print("Please run the script like this:")
        print("OOD_WEATHER=Fog OOD_SEVERITY=easy DATE_TIME_STAMP=20260501_000617 python safety_monitor/energy_score.py\n")
        sys.exit(1)

    target_dir = os.path.join(LOGIT_OUTPUT, weather, severity, timestamp)

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
    print(f"Successfully calculated energy scores for {len(frame_results)} frames.")
    
    print("\n--- Sample Results ---")
    for idx, (token, score) in enumerate(frame_results.items()):
        if idx >= 5:
            break
        print(f"Token: {token} | Energy: {score:.4f}")

    model_state_dict = torch.load(file_path)
    print("Model dict:", model_state_dict)
    print("Shape of Model dict:", model_state_dict.shape)

    energy_score = compute_energy_score(model_state_dict)
    print("Energy score", energy_score)
    print("Shape of energy score:", energy_score.shape)

