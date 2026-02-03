"""
Physics-Constrained Temporal Fusion Transformer.

This module implements a modified TFT architecture that incorporates
physical constraints into the temporal prediction framework.

Key Modifications from Standard TFT
-----------------------------------
1. Physics-derived features as static enrichment
2. Constraint-aware attention mechanisms
3. Output heads with physical bounds
4. Residual physics connections

References
----------
- Lim, B. et al. (2021). Temporal Fusion Transformers for Interpretable
  Multi-horizon Time Series Forecasting. IJCAI.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class TFTConfig:
    """Configuration for Physics-Constrained TFT.
    
    Attributes
    ----------
    hidden_dim : int
        Hidden dimension throughout the model.
    num_attention_heads : int
        Number of attention heads.
    num_encoder_layers : int
        Number of transformer encoder layers.
    num_decoder_layers : int
        Number of transformer decoder layers.
    dropout : float
        Dropout probability.
    forecast_horizon : int
        Number of future time steps to predict.
    lookback_horizon : int
        Number of past time steps to use.
    physics_feature_dim : int
        Dimension of physics-derived features.
    quantile_outputs : list
        Quantiles to predict for uncertainty.
    """
    hidden_dim: int = 256
    num_attention_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout: float = 0.1
    forecast_horizon: int = 10  # 60 hours at 6h intervals
    lookback_horizon: int = 8   # 48 hours of history
    physics_feature_dim: int = 32
    quantile_outputs: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 0.9]
    )


class GatedLinearUnit(nn.Module):
    """Gated Linear Unit for TFT."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.sigmoid(self.fc1(x)) * self.fc2(x))


class VariableSelectionNetwork(nn.Module):
    """Variable Selection Network for identifying important features."""
    
    def __init__(
        self,
        input_dim: int,
        num_inputs: int,
        hidden_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_inputs = num_inputs
        
        # Flattened GRN for variable selection weights
        self.flattened_gru = nn.Sequential(
            nn.Linear(input_dim * num_inputs, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_inputs),
            nn.Softmax(dim=-1)
        )
        
        # Per-variable transformations
        self.variable_transforms = nn.ModuleList([
            GatedLinearUnit(input_dim, hidden_dim, dropout)
            for _ in range(num_inputs)
        ])
    
    def forward(self, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        inputs : List[torch.Tensor]
            List of input tensors, each of shape (B, T, D).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (combined_output, variable_weights)
        """
        # Flatten inputs for weight computation
        flattened = torch.cat(inputs, dim=-1)
        weights = self.flattened_gru(flattened)  # (B, T, num_inputs)
        
        # Transform each variable
        transformed = torch.stack([
            self.variable_transforms[i](inputs[i])
            for i in range(self.num_inputs)
        ], dim=-1)  # (B, T, hidden_dim, num_inputs)
        
        # Weighted combination
        weights_expanded = weights.unsqueeze(-2)  # (B, T, 1, num_inputs)
        combined = (transformed * weights_expanded).sum(dim=-1)
        
        return combined, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Multi-head attention with interpretable attention weights."""
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (output, attention_weights)
        """
        B, T, _ = query.shape
        
        # Project
        q = self.q_proj(query).view(B, T, self.num_heads, self.head_dim)
        k = self.k_proj(key).view(B, -1, self.num_heads, self.head_dim)
        v = self.v_proj(value).view(B, -1, self.num_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, -1)
        
        output = self.out_proj(attn_output)
        
        # Average attention weights across heads for interpretability
        avg_attn = attn_weights.mean(dim=1)
        
        return output, avg_attn


class PhysicsEnrichmentLayer(nn.Module):
    """Layer that enriches representations with physics-derived features.
    
    This layer combines learned temporal features with physically
    computed quantities (steering, beta drift, potential intensity).
    """
    
    def __init__(
        self,
        hidden_dim: int,
        physics_dim: int,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.physics_projection = nn.Linear(physics_dim, hidden_dim)
        self.gate = GatedLinearUnit(hidden_dim * 2, hidden_dim, dropout)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(
        self,
        temporal_features: torch.Tensor,
        physics_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        temporal_features : torch.Tensor
            Learned temporal features (B, T, hidden_dim).
        physics_features : torch.Tensor
            Physics-derived features (B, T, physics_dim).
            
        Returns
        -------
        torch.Tensor
            Enriched features (B, T, hidden_dim).
        """
        physics_projected = self.physics_projection(physics_features)
        combined = torch.cat([temporal_features, physics_projected], dim=-1)
        gated = self.gate(combined)
        return self.layer_norm(temporal_features + gated)


class PhysicsConstrainedTFT(nn.Module):
    """Physics-Constrained Temporal Fusion Transformer.
    
    This model combines the interpretable architecture of TFT with
    physics-based constraints for cyclone trajectory prediction.
    
    Architecture
    ------------
    1. Variable Selection Networks for input processing
    2. Physics Enrichment Layers for constraint integration
    3. Interpretable Multi-Head Attention for temporal patterns
    4. Quantile outputs for uncertainty estimation
    
    Attributes
    ----------
    config : TFTConfig
        Model configuration.
    """
    
    def __init__(self, config: TFTConfig):
        super().__init__()
        
        self.config = config
        self._logger = get_logger("PhysicsConstrainedTFT")
        
        # Input embeddings
        self.static_embedding = nn.Linear(32, config.hidden_dim)
        self.temporal_embedding = nn.Linear(64, config.hidden_dim)
        
        # Variable selection
        self.encoder_vsn = VariableSelectionNetwork(
            input_dim=config.hidden_dim,
            num_inputs=4,  # Placeholder
            hidden_dim=config.hidden_dim,
            dropout=config.dropout
        )
        
        # Physics enrichment
        self.physics_enrichment = PhysicsEnrichmentLayer(
            hidden_dim=config.hidden_dim,
            physics_dim=config.physics_feature_dim,
            dropout=config.dropout
        )
        
        # Encoder layers
        self.encoder_attention = nn.ModuleList([
            InterpretableMultiHeadAttention(
                config.hidden_dim,
                config.num_attention_heads,
                config.dropout
            )
            for _ in range(config.num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_attention = nn.ModuleList([
            InterpretableMultiHeadAttention(
                config.hidden_dim,
                config.num_attention_heads,
                config.dropout
            )
            for _ in range(config.num_decoder_layers)
        ])
        
        # Output heads
        num_quantiles = len(config.quantile_outputs)
        
        # Position output (lat, lon)
        self.position_head = nn.Linear(config.hidden_dim, 2 * num_quantiles)
        
        # Intensity output (max wind)
        self.intensity_head = nn.Linear(config.hidden_dim, num_quantiles)
        
        # Pressure output (central pressure)
        self.pressure_head = nn.Linear(config.hidden_dim, num_quantiles)
    
    def forward(
        self,
        past_observed: torch.Tensor,
        past_physics: torch.Tensor,
        static_features: torch.Tensor,
        future_physics: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Parameters
        ----------
        past_observed : torch.Tensor
            Past observed features (B, lookback, feature_dim).
        past_physics : torch.Tensor
            Past physics-derived features (B, lookback, physics_dim).
        static_features : torch.Tensor
            Static covariates (B, static_dim).
        future_physics : torch.Tensor, optional
            Future physics features if available (B, horizon, physics_dim).
            
        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys:
            - 'position': (B, horizon, 2, num_quantiles)
            - 'intensity': (B, horizon, num_quantiles)
            - 'pressure': (B, horizon, num_quantiles)
            - 'attention_weights': attention maps for interpretability
        """
        B = past_observed.shape[0]
        
        # Static embedding
        static_encoded = self.static_embedding(static_features)
        static_expanded = static_encoded.unsqueeze(1)
        
        # Temporal embedding
        temporal_encoded = self.temporal_embedding(past_observed)
        
        # Simple VSN application (placeholder for full implementation)
        # In full implementation, would use proper variable selection
        encoder_input = temporal_encoded
        
        # Physics enrichment
        enriched = self.physics_enrichment(encoder_input, past_physics)
        
        # Encoder
        encoder_output = enriched
        encoder_attentions = []
        
        for attention_layer in self.encoder_attention:
            attn_out, attn_weights = attention_layer(
                encoder_output, encoder_output, encoder_output
            )
            encoder_output = encoder_output + attn_out
            encoder_attentions.append(attn_weights)
        
        # Decoder (autoregressive generation)
        horizon = self.config.forecast_horizon
        decoder_output = encoder_output[:, -1:, :].expand(-1, horizon, -1)
        
        # If future physics available, incorporate
        if future_physics is not None:
            decoder_output = self.physics_enrichment(decoder_output, future_physics)
        
        decoder_attentions = []
        for attention_layer in self.decoder_attention:
            attn_out, attn_weights = attention_layer(
                decoder_output, encoder_output, encoder_output
            )
            decoder_output = decoder_output + attn_out
            decoder_attentions.append(attn_weights)
        
        # Output heads
        num_q = len(self.config.quantile_outputs)
        
        position = self.position_head(decoder_output)
        position = position.view(B, horizon, 2, num_q)
        
        intensity = self.intensity_head(decoder_output)
        intensity = intensity.view(B, horizon, num_q)
        
        pressure = self.pressure_head(decoder_output)
        pressure = pressure.view(B, horizon, num_q)
        
        # Apply physical constraints to outputs
        intensity = F.relu(intensity)  # Wind must be positive
        pressure = 1013 - F.relu(1013 - pressure)  # Pressure <= 1013 hPa
        
        return {
            'position': position,
            'intensity': intensity,
            'pressure': pressure,
            'encoder_attention': torch.stack(encoder_attentions, dim=1),
            'decoder_attention': torch.stack(decoder_attentions, dim=1),
        }
    
    def get_interpretable_outputs(
        self,
        outputs: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """Extract interpretable information from model outputs.
        
        Returns attention patterns and uncertainty estimates that can
        be used for model interpretation and validation.
        """
        return {
            'temporal_attention': outputs['encoder_attention'].mean(dim=1),
            'forecast_attention': outputs['decoder_attention'].mean(dim=1),
            'position_uncertainty': (
                outputs['position'][:, :, :, -1] - 
                outputs['position'][:, :, :, 0]
            ),
            'intensity_uncertainty': (
                outputs['intensity'][:, :, -1] - 
                outputs['intensity'][:, :, 0]
            ),
        }
