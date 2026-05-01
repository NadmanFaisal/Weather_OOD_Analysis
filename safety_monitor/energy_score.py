import torch
import os

current_dir = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.abspath(os.path.join(
    current_dir,
    "../",
    "data/intercepted_logits/Fog/easy/20260501_000617/head_final_logits_batch_0_gpu_57155.pt"
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
    scores = -T * torch.logsumexp(logits / T, dim=1)
    return scores.detach()

if __name__ == "__main__":




    model_state_dict = torch.load(file_path)
    print("Model dict:", model_state_dict)
    print("Shape of Model dict:", model_state_dict.shape)

    energy_score = compute_energy_score(model_state_dict)
    print("Energy score", energy_score)
    print("Shape of energy score:", energy_score.shape)
