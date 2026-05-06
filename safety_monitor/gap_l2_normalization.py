import torch
import torch.nn.functional as F

def apply_feature_regularization(features: torch.Tensor, use_gap: bool = True, use_l2: bool = True) -> torch.Tensor:
    
    if use_gap:
        features = features.mean(dim=0, keepdim=True)

    if use_l2:
        features = F.normalize(features, p=2, dim=-1)

    return features
