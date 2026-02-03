"""
Multi-Branch Fusion Layer for Cyclone Prediction.

This module combines outputs from different model branches:
1. Vision branch (spatial features from satellite)
2. Temporal branch (TFT predictions)
3. Physics branch (physical calculations)

The fusion respects the different nature of each branch:
- Vision: learned patterns without assumed physical meaning
- Temporal: statistical predictions with uncertainty
- Physics: deterministic calculations with known validity

References
----------
- Bi, K. et al. (2022). Pangu-Weather. arXiv.
- Pathak, J. et al. (2022). FourCastNet. arXiv.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class FusionConfig:
    """Configuration for multi-branch fusion.
    
    Attributes
    ----------
    vision_dim : int
        Dimension of vision branch features.
    temporal_dim : int
        Dimension of temporal branch features.
    physics_dim : int
        Dimension of physics branch features.
    fusion_dim : int
        Dimension of fused representation.
    num_heads : int
        Number of attention heads for cross-modal fusion.
    dropout : float
        Dropout probability.
    physics_weight : float
        Relative weight for physics branch (higher = more physics trust).
    """
    vision_dim: int = 256
    temporal_dim: int = 256
    physics_dim: int = 32
    fusion_dim: int = 256
    num_heads: int = 8
    dropout: float = 0.1
    physics_weight: float = 0.3


class CrossModalAttention(nn.Module):
    """Cross-modal attention for fusing different branches.
    
    Uses one branch as query to attend over another branch,
    enabling selective information fusion.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        output_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        self.q_proj = nn.Linear(query_dim, output_dim)
        self.k_proj = nn.Linear(key_dim, output_dim)
        self.v_proj = nn.Linear(key_dim, output_dim)
        self.out_proj = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        query : torch.Tensor
            Query tensor (B, T_q, query_dim).
        key_value : torch.Tensor
            Key/value tensor (B, T_kv, key_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (output, attention_weights)
        """
        B, T_q, _ = query.shape
        T_kv = key_value.shape[1]
        
        # Project
        q = self.q_proj(query).view(B, T_q, self.num_heads, self.head_dim)
        k = self.k_proj(key_value).view(B, T_kv, self.num_heads, self.head_dim)
        v = self.v_proj(key_value).view(B, T_kv, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(B, T_q, -1)
        output = self.out_proj(output)
        
        return self.layer_norm(output), attn_weights.mean(dim=1)


class PhysicsGatingLayer(nn.Module):
    """Gating layer that modulates features based on physics consistency.
    
    This layer learns to trust data-driven features when they are
    consistent with physics, and fall back to physics when they diverge.
    """
    
    def __init__(
        self,
        feature_dim: int,
        physics_dim: int,
        base_physics_weight: float = 0.3
    ):
        super().__init__()
        
        self.base_physics_weight = base_physics_weight
        
        # Consistency scoring
        self.consistency_scorer = nn.Sequential(
            nn.Linear(feature_dim + physics_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Physics projection to feature dim
        self.physics_proj = nn.Linear(physics_dim, feature_dim)
    
    def forward(
        self,
        features: torch.Tensor,
        physics: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        features : torch.Tensor
            Data-driven features (B, T, feature_dim).
        physics : torch.Tensor
            Physics-derived features (B, T, physics_dim).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (gated_features, consistency_score)
        """
        # Compute consistency between features and physics
        combined = torch.cat([features, physics], dim=-1)
        consistency = self.consistency_scorer(combined)
        
        # Adaptive physics weight: more physics when inconsistent
        # consistency near 1 = consistent, use less physics override
        # consistency near 0 = inconsistent, use more physics
        adaptive_weight = self.base_physics_weight + (1 - consistency) * 0.4
        
        # Project physics to feature dimension
        physics_projected = self.physics_proj(physics)
        
        # Weighted combination
        gated = (1 - adaptive_weight) * features + adaptive_weight * physics_projected
        
        return gated, consistency.squeeze(-1)


class MultiBranchFusion(nn.Module):
    """Multi-branch fusion layer combining vision, temporal, and physics.
    
    Architecture
    ------------
    1. Cross-modal attention: vision attends to temporal
    2. Physics gating: modulate based on physics consistency
    3. Final fusion: combine into unified representation
    
    The physics branch serves as a regularizer and fallback when
    data-driven predictions are uncertain or inconsistent.
    """
    
    def __init__(self, config: FusionConfig):
        super().__init__()
        
        self.config = config
        self._logger = get_logger("MultiBranchFusion")
        
        # Project branches to common dimension
        self.vision_proj = nn.Linear(config.vision_dim, config.fusion_dim)
        self.temporal_proj = nn.Linear(config.temporal_dim, config.fusion_dim)
        self.physics_proj = nn.Linear(config.physics_dim, config.fusion_dim)
        
        # Cross-modal attention: temporal queries vision
        self.temporal_vision_attn = CrossModalAttention(
            query_dim=config.fusion_dim,
            key_dim=config.fusion_dim,
            output_dim=config.fusion_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Physics gating
        self.physics_gate = PhysicsGatingLayer(
            feature_dim=config.fusion_dim,
            physics_dim=config.physics_dim,
            base_physics_weight=config.physics_weight
        )
        
        # Final fusion
        self.fusion_mlp = nn.Sequential(
            nn.Linear(config.fusion_dim * 2, config.fusion_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.fusion_dim, config.fusion_dim),
        )
        
        self.layer_norm = nn.LayerNorm(config.fusion_dim)
    
    def forward(
        self,
        vision_features: torch.Tensor,
        temporal_features: torch.Tensor,
        physics_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        vision_features : torch.Tensor
            Features from vision branch (B, H*W, vision_dim).
        temporal_features : torch.Tensor
            Features from temporal branch (B, T, temporal_dim).
        physics_features : torch.Tensor
            Features from physics branch (B, T, physics_dim).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with:
            - 'fused': Final fused features (B, T, fusion_dim)
            - 'vision_attention': Attention over vision features
            - 'physics_consistency': Consistency scores
        """
        # Project to common dimension
        vision_proj = self.vision_proj(vision_features)
        temporal_proj = self.temporal_proj(temporal_features)
        physics_proj_full = self.physics_proj(physics_features)
        
        # Cross-modal attention: temporal queries vision
        temporal_with_vision, vision_attn = self.temporal_vision_attn(
            temporal_proj, vision_proj
        )
        
        # Combine temporal with vision
        combined = temporal_proj + temporal_with_vision
        
        # Physics gating
        gated, consistency = self.physics_gate(combined, physics_features)
        
        # Final fusion
        fused_input = torch.cat([gated, physics_proj_full], dim=-1)
        fused = self.fusion_mlp(fused_input)
        fused = self.layer_norm(fused + gated)
        
        return {
            'fused': fused,
            'vision_attention': vision_attn,
            'physics_consistency': consistency,
        }
