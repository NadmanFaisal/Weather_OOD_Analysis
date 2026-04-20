import math
import torch
import pytest
from safety_monitor.energy_score import compute_energy_score


def test_output_shape():
    # Single sample: output should be a 1-D tensor of length 1
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    scores = compute_energy_score(logits, T=1.0)
    assert scores.shape == (1,)


def test_known_value():
    # E = -1.0 * log(exp(1.0) + exp(2.0) + exp(0.5))
    logits = torch.tensor([[1.0, 2.0, 0.5]])
    scores = compute_energy_score(logits, T=1.0)
    expected = -math.log(math.exp(1.0) + math.exp(2.0) + math.exp(0.5))
    assert abs(scores[0].item() - expected) < 1e-5


def test_no_gradient_retained():
    # .detach() must be called so the scoring step doesn't retain grad info
    logits = torch.tensor([[1.0, 2.0, 0.5]], requires_grad=True)
    scores = compute_energy_score(logits, T=1.0)
    assert not scores.requires_grad


def test_peaked_lower_energy_than_flat():
    # Peaked logits (in-distribution proxy) should have lower energy than flat/uniform (OOD proxy)
    peaked = torch.tensor([[10.0, 0.1, 0.1, 0.1]])
    flat = torch.tensor([[2.5, 2.5, 2.5, 2.5]])
    peaked_score = compute_energy_score(peaked, T=1.0)
    flat_score = compute_energy_score(flat, T=1.0)
    assert peaked_score.item() < flat_score.item()


def test_batch_output_shape():
    # Batch of 8 samples with 10 classes: output should be shape (8,)
    logits = torch.randn(8, 10)
    scores = compute_energy_score(logits, T=1.0)
    assert scores.shape == (8,)
