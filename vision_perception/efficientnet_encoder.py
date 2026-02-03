"""
EfficientNet Encoder for Satellite Image Feature Extraction.

This module provides a spatial pattern extractor based on EfficientNet.
It is treated strictly as a texture/pattern extractor - NO physical
meaning is assumed from the raw outputs.

Critical Limitation
-------------------
The features extracted by this encoder are visual representations only.
They do not directly correspond to physical quantities. Any physical
interpretation must be done in downstream modules with appropriate
calibration and validation.

References
----------
- Tan, M. & Le, Q.V. (2019). EfficientNet: Rethinking model scaling for CNNs.
"""

from typing import Optional, Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.logging_config import get_logger

logger = get_logger(__name__)


class SatelliteImageProcessor:
    """Preprocessor for satellite images before neural network input.
    
    This class handles the conversion from geophysical data to
    network-ready tensor format, including:
    - Channel stacking for multi-spectral data
    - Normalization using satellite-specific statistics
    - Handling of missing data (clouds, sensor gaps)
    
    Attributes
    ----------
    channels : list
        List of channel names (e.g., ['IR', 'VIS', 'WV']).
    normalization_stats : dict
        Per-channel mean and std for normalization.
    """
    
    # Typical normalization statistics for geostationary satellite data
    DEFAULT_STATS = {
        'IR': {'mean': 270.0, 'std': 30.0},  # Kelvin
        'VIS': {'mean': 0.3, 'std': 0.25},    # Reflectance
        'WV': {'mean': 240.0, 'std': 20.0},   # Kelvin
    }
    
    def __init__(
        self,
        channels: List[str],
        normalization_stats: Optional[Dict[str, Dict[str, float]]] = None,
        fill_value: float = 0.0
    ):
        """Initialize processor.
        
        Parameters
        ----------
        channels : list
            Channel names to process.
        normalization_stats : dict, optional
            Per-channel mean and std. Uses defaults if not provided.
        fill_value : float
            Value to use for missing data after normalization.
        """
        self.channels = channels
        self.normalization_stats = normalization_stats or self.DEFAULT_STATS
        self.fill_value = fill_value
        self._logger = get_logger("SatelliteImageProcessor")
    
    def process(
        self,
        data: Dict[str, NDArray[np.float32]],
        mask: Optional[NDArray[np.bool_]] = None
    ) -> torch.Tensor:
        """Process satellite data for neural network input.
        
        Parameters
        ----------
        data : dict
            Dictionary mapping channel names to 2D arrays.
        mask : ndarray, optional
            Boolean mask where True indicates invalid data.
            
        Returns
        -------
        torch.Tensor
            Tensor of shape (C, H, W) ready for network input.
        """
        processed_channels = []
        
        for channel in self.channels:
            if channel not in data:
                raise ValueError(f"Missing channel: {channel}")
            
            channel_data = data[channel].astype(np.float32)
            
            # Get normalization stats
            stats = self.normalization_stats.get(channel, {'mean': 0, 'std': 1})
            
            # Normalize
            normalized = (channel_data - stats['mean']) / stats['std']
            
            # Handle missing data
            if mask is not None:
                normalized[mask] = self.fill_value
            
            processed_channels.append(normalized)
        
        # Stack channels
        stacked = np.stack(processed_channels, axis=0)
        
        return torch.from_numpy(stacked)


class EfficientNetEncoder(nn.Module):
    """EfficientNet-based encoder for satellite imagery.
    
    This encoder extracts hierarchical spatial features from satellite
    images. It outputs a multi-scale feature pyramid suitable for
    both local (eye structure) and large-scale (outer bands) patterns.
    
    IMPORTANT LIMITATION
    --------------------
    This is a PATTERN EXTRACTOR. The output features are learned
    representations that capture visual patterns, NOT physical quantities.
    Do not interpret feature values as having physical meaning without
    explicit calibration.
    
    Attributes
    ----------
    backbone : str
        EfficientNet variant used (e.g., 'efficientnet_b4').
    in_channels : int
        Number of input channels.
    feature_dims : list
        Dimensions of features at each pyramid level.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        backbone: str = 'efficientnet_b4',
        pretrained: bool = True,
        freeze_bn: bool = False
    ):
        """Initialize encoder.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels (satellite bands).
        backbone : str
            EfficientNet variant to use.
        pretrained : bool
            Whether to use ImageNet pretrained weights.
        freeze_bn : bool
            Whether to freeze batch normalization layers.
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.backbone_name = backbone
        self._logger = get_logger("EfficientNetEncoder")
        
        # Import timm for EfficientNet
        try:
            import timm
        except ImportError:
            raise ImportError("Please install timm: pip install timm")
        
        # Create backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            in_chans=in_channels
        )
        
        # Get feature dimensions from backbone
        self.feature_dims = self.backbone.feature_info.channels()
        
        self._logger.info(
            f"Initialized {backbone} with {in_channels} input channels, "
            f"feature dims: {self.feature_dims}"
        )
        
        if freeze_bn:
            self._freeze_bn()
    
    def _freeze_bn(self):
        """Freeze batch normalization layers."""
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                module.eval()
                for param in module.parameters():
                    param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale features.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W).
            
        Returns
        -------
        List[torch.Tensor]
            List of feature maps at different scales.
            Scales are ordered from finest (highest resolution) to
            coarsest (lowest resolution).
            
        Notes
        -----
        These features are VISUAL PATTERNS, not physical quantities.
        """
        features = self.backbone(x)
        return features
    
    def get_attention_maps(
        self,
        x: torch.Tensor,
        layer_idx: int = -1
    ) -> torch.Tensor:
        """Compute attention maps for interpretability.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        layer_idx : int
            Which feature layer to compute attention for.
            
        Returns
        -------
        torch.Tensor
            Attention map of shape (B, H, W).
        """
        features = self.forward(x)
        feature_map = features[layer_idx]
        
        # Global average across channels
        attention = feature_map.mean(dim=1)
        
        # Normalize to [0, 1]
        attention = attention - attention.min()
        attention = attention / (attention.max() + 1e-8)
        
        return attention
    
    @property
    def output_dim(self) -> int:
        """Total feature dimension if flattened."""
        return sum(self.feature_dims)


class FeaturePyramidNetwork(nn.Module):
    """Feature Pyramid Network for multi-scale fusion.
    
    Combines features from different scales of the encoder into
    a unified representation.
    """
    
    def __init__(
        self,
        in_channels_list: List[int],
        out_channels: int = 256
    ):
        """Initialize FPN.
        
        Parameters
        ----------
        in_channels_list : list
            Number of channels at each input scale.
        out_channels : int
            Output channels at all scales.
        """
        super().__init__()
        
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_ch, out_channels, 1)
            for in_ch in in_channels_list
        ])
        
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in in_channels_list
        ])
        
        self.out_channels = out_channels
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Build feature pyramid.
        
        Parameters
        ----------
        features : list
            Features from encoder at different scales.
            
        Returns
        -------
        list
            FPN features at each scale.
        """
        # Apply lateral convolutions
        laterals = [
            lateral_conv(features[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        
        # Top-down pathway
        for i in range(len(laterals) - 1, 0, -1):
            size = laterals[i - 1].shape[2:]
            upsampled = F.interpolate(
                laterals[i], size=size, mode='nearest'
            )
            laterals[i - 1] = laterals[i - 1] + upsampled
        
        # Output convolutions
        outputs = [
            output_conv(laterals[i])
            for i, output_conv in enumerate(self.output_convs)
        ]
        
        return outputs
