import torch
import pytest
from safety_monitor.mahalanobis import MahalanobisDetector


def test_output_shape():
    detector = MahalanobisDetector()
    torch.manual_seed(0)
    features = torch.randn(3, 4)
    detector.fit(features)
    scores = detector.score(features)
    assert scores.shape == (3,), f"Expected shape (3,), got {scores.shape}"


def test_manual_value():
    """Verify score against a manually computed Mahalanobis distance."""
    features = torch.tensor([[1.0, 0.0],
                              [0.0, 1.0],
                              [1.0, 1.0]], dtype=torch.float64)
    detector = MahalanobisDetector()
    detector.fit(features)

    # Manually compute expected distance for the first sample
    mean = features.mean(dim=0)
    centered = features - mean
    cov = (centered.T @ centered) / 2
    cov = cov + 1e-6 * torch.eye(2, dtype=torch.float64)
    inv_cov = torch.linalg.inv(cov)
    delta = features[0] - mean
    expected = float(torch.sqrt(delta @ inv_cov @ delta))

    scores = detector.score(features)
    assert abs(float(scores[0]) - expected) < 1e-5, (
        f"Expected {expected:.6f}, got {float(scores[0]):.6f}"
    )


def test_no_gradient():
    detector = MahalanobisDetector()
    torch.manual_seed(1)
    features = torch.randn(3, 4, requires_grad=False)
    detector.fit(features)
    scores = detector.score(features)
    assert not scores.requires_grad, "Output tensor must not retain gradients"


def test_id_lower_than_ood():
    """In-distribution samples should score lower than clearly OOD samples."""
    torch.manual_seed(0)
    # Fit on tight cluster around origin
    id_train = torch.randn(50, 4) * 0.1
    detector = MahalanobisDetector()
    detector.fit(id_train)

    id_test = torch.randn(10, 4) * 0.1
    ood_test = torch.randn(10, 4) * 10.0 + 50.0  # far from origin

    id_scores = detector.score(id_test)
    ood_scores = detector.score(ood_test)

    assert id_scores.mean() < ood_scores.mean(), (
        f"ID mean {id_scores.mean():.2f} should be less than OOD mean {ood_scores.mean():.2f}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
