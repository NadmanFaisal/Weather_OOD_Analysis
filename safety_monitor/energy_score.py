import torch


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
