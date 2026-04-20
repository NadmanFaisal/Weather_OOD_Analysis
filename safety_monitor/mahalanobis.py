import torch


class MahalanobisDetector:
    """OOD detector using Mahalanobis distance on latent features.

    Real features will be injected by a PyTorch forward hook in a later issue.
    Currently accepts dummy/synthetic feature tensors for development and testing.
    """

    def __init__(self):
        self.mean = None
        self.inv_cov = None

    def fit(self, features: torch.Tensor) -> None:
        """Compute and store the mean and inverse covariance from training features.

        Args:
            features: (N, D) tensor of in-distribution feature vectors.

        Note: real features will be injected by a PyTorch hook in a later issue.
        """
        mean = features.mean(dim=0)
        centered = features - mean
        cov = (centered.T @ centered) / (features.shape[0] - 1)
        # Small regularisation to prevent singular covariance for low-rank inputs
        cov = cov + 1e-6 * torch.eye(features.shape[1], dtype=features.dtype)
        self.mean = mean
        self.inv_cov = torch.linalg.inv(cov)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """Compute Mahalanobis distance for each feature vector.

        Args:
            features: (N, D) tensor of feature vectors to score.

        Returns:
            scores: (N,) tensor of Mahalanobis distances (detached, no gradients).

        Note: real features will be injected by a PyTorch hook in a later issue.
        """
        if self.mean is None or self.inv_cov is None:
            raise RuntimeError("Call fit() before score().")
        delta = features - self.mean
        scores = torch.sqrt((delta @ self.inv_cov * delta).sum(dim=1))
        return scores.detach()
