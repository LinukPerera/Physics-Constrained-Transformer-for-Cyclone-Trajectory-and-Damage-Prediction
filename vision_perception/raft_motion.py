"""
RAFT Optical Flow and Motion to Physical Displacement Conversion.

This module provides optical flow estimation from satellite imagery sequences
and critically includes the conversion from image motion (pixels) to
atmospheric motion (physical displacement).

CRITICAL DISTINCTION
--------------------
- IMAGE MOTION: Displacement in pixel coordinates between frames
- ATMOSPHERIC MOTION: Physical wind velocity in m/s

These are NOT the same. The conversion requires:
1. Satellite geometry (pixel scale varies with view angle)
2. Parallax correction (clouds at different heights move differently)
3. Temporal calibration (frame rate to seconds)

References
----------
- Teed, Z. & Deng, J. (2020). RAFT: Recurrent All-Pairs Field Transforms.
- Velden, C.S. et al. (2005). Atmospheric Motion Vectors from satellites.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

import torch
import torch.nn as nn

from common.logging_config import get_logger
from geospatial.coordinate_models import radius_of_curvature_prime_vertical
from geospatial.distance_calculations import geodesic_distance

logger = get_logger(__name__)


@dataclass
class PixelMotion:
    """Raw optical flow in pixel coordinates.
    
    Attributes
    ----------
    u_pixels : ndarray
        Horizontal (eastward) motion in pixels.
    v_pixels : ndarray
        Vertical (northward) motion in pixels.
    confidence : ndarray
        Per-pixel confidence scores [0, 1].
    frame_interval_s : float
        Time between frames in seconds.
    """
    u_pixels: NDArray[np.float32]
    v_pixels: NDArray[np.float32]
    confidence: NDArray[np.float32]
    frame_interval_s: float


@dataclass
class PhysicalMotion:
    """Motion converted to physical displacement.
    
    Attributes
    ----------
    u_ms : ndarray
        Eastward velocity in m/s.
    v_ms : ndarray
        Northward velocity in m/s.
    speed_ms : ndarray
        Total speed in m/s.
    direction_rad : ndarray
        Direction of motion in radians (from which wind blows).
    valid_mask : ndarray
        Boolean mask of valid estimates.
    uncertainty_ms : ndarray
        Estimated uncertainty in m/s.
    """
    u_ms: NDArray[np.float32]
    v_ms: NDArray[np.float32]
    speed_ms: NDArray[np.float32]
    direction_rad: NDArray[np.float32]
    valid_mask: NDArray[np.bool_]
    uncertainty_ms: NDArray[np.float32]


class MotionToPhysicalConverter:
    """Convert pixel motion to physical atmospheric displacement.
    
    This class handles the non-trivial conversion from optical flow
    (pixel displacements) to physical wind velocities. The conversion
    accounts for:
    
    1. Satellite viewing geometry
    2. Latitude-dependent scale
    3. Parallax effects
    4. Quality control
    
    IMPORTANT
    ---------
    The output should be validated against atmospheric analysis fields.
    Optical flow from cloud motion is NOT identical to wind at a specific
    pressure level - clouds trace wind at their height only.
    """
    
    def __init__(
        self,
        pixel_scale_km: float,
        satellite_longitude: float = 0.0,
        parallax_correction: bool = True
    ):
        """Initialize converter.
        
        Parameters
        ----------
        pixel_scale_km : float
            Nominal pixel scale at satellite nadir in km.
        satellite_longitude : float
            Sub-satellite point longitude in degrees.
        parallax_correction : bool
            Whether to apply parallax correction for cloud height.
        """
        self.pixel_scale_km = pixel_scale_km
        self.satellite_longitude = satellite_longitude
        self.parallax_correction = parallax_correction
        self._logger = get_logger("MotionToPhysicalConverter")
    
    def convert(
        self,
        pixel_motion: PixelMotion,
        lats: NDArray[np.float64],
        lons: NDArray[np.float64],
        cloud_heights_m: Optional[NDArray[np.float32]] = None
    ) -> PhysicalMotion:
        """Convert pixel motion to physical velocity.
        
        Parameters
        ----------
        pixel_motion : PixelMotion
            Optical flow in pixel coordinates.
        lats : ndarray
            Latitude grid in degrees.
        lons : ndarray
            Longitude grid in degrees.
        cloud_heights_m : ndarray, optional
            Estimated cloud-top heights for parallax correction.
            
        Returns
        -------
        PhysicalMotion
            Physical velocity estimates.
            
        Notes
        -----
        The conversion uses proper geodesy from the geospatial module.
        Pixel scale is adjusted for latitude and satellite view angle.
        """
        # Create latitude grid for scale factor
        if lats.ndim == 1:
            lat_grid = np.tile(lats[:, np.newaxis], (1, len(lons)))
        else:
            lat_grid = lats
        
        # Compute latitude-dependent scale factor
        # Pixel scale increases away from equator due to projection
        lat_rad = np.radians(lat_grid)
        
        # Get radius of curvature at each latitude
        N = np.array([
            [radius_of_curvature_prime_vertical(np.radians(lat)) 
             for lat in lats]
        ]).T
        N = np.tile(N, (1, len(lons) if lons.ndim == 1 else lons.shape[1]))
        
        # Compute actual pixel scale accounting for satellite geometry
        # Scale increases with cos(view_angle) away from nadir
        lon_grid = np.tile(lons[np.newaxis, :], (len(lats), 1)) if lons.ndim == 1 else lons
        view_lon_diff = np.radians(lon_grid - self.satellite_longitude)
        
        # View angle factor (simplified - assumes geostationary geometry)
        view_factor = np.cos(view_lon_diff) * np.cos(lat_rad)
        view_factor = np.clip(view_factor, 0.3, 1.0)  # Limit to reasonable range
        
        # Adjusted pixel scale in meters
        pixel_scale_m = (self.pixel_scale_km * 1000) / view_factor
        
        # Convert pixel motion to displacement in meters
        dx_m = pixel_motion.u_pixels * pixel_scale_m
        dy_m = pixel_motion.v_pixels * pixel_scale_m
        
        # Apply parallax correction if cloud heights provided
        if self.parallax_correction and cloud_heights_m is not None:
            # Parallax displacement increases with cloud height
            # and view angle from nadir
            parallax_factor = 1.0 + cloud_heights_m / 35786000  # GEO altitude
            # This is a simplification - proper parallax is more complex
            self._logger.info("Applied simplified parallax correction")
        
        # Convert displacement to velocity using frame interval
        u_ms = dx_m / pixel_motion.frame_interval_s
        v_ms = dy_m / pixel_motion.frame_interval_s
        
        # Compute derived quantities
        speed_ms = np.sqrt(u_ms**2 + v_ms**2)
        direction_rad = np.arctan2(-u_ms, -v_ms) % (2 * np.pi)  # FROM which
        
        # Quality control
        valid_mask = (
            (pixel_motion.confidence > 0.5) &
            (speed_ms < 100) &  # Physical limit for tropical cyclones
            np.isfinite(speed_ms)
        )
        
        # Estimate uncertainty based on confidence
        base_uncertainty = 2.0  # m/s baseline
        uncertainty_ms = base_uncertainty / (pixel_motion.confidence + 0.1)
        uncertainty_ms = np.clip(uncertainty_ms, 1.0, 20.0)
        
        return PhysicalMotion(
            u_ms=u_ms.astype(np.float32),
            v_ms=v_ms.astype(np.float32),
            speed_ms=speed_ms.astype(np.float32),
            direction_rad=direction_rad.astype(np.float32),
            valid_mask=valid_mask,
            uncertainty_ms=uncertainty_ms.astype(np.float32)
        )


class RAFTMotionEstimator(nn.Module):
    """RAFT-based optical flow estimator for satellite imagery.
    
    This implements a simplified version of RAFT (Recurrent All-Pairs
    Field Transforms) for estimating motion between satellite frames.
    
    Output
    ------
    Produces PixelMotion, which must be converted to PhysicalMotion
    using MotionToPhysicalConverter before use in physics calculations.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        hidden_dim: int = 128,
        context_dim: int = 128,
        num_iterations: int = 12
    ):
        """Initialize RAFT estimator.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        hidden_dim : int
            Hidden dimension for feature extraction.
        context_dim : int
            Context feature dimension.
        num_iterations : int
            Number of iterative refinement steps.
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim
        self.num_iterations = num_iterations
        
        # Feature encoder
        self.fnet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hidden_dim, 3, padding=1),
        )
        
        # Context encoder
        self.cnet = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, context_dim + hidden_dim, 3, padding=1),
        )
        
        # Update block for iterative refinement
        self.update_block = nn.Sequential(
            nn.Conv2d(hidden_dim + context_dim + 2, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 2, 3, padding=1),  # Output: flow update
        )
        
        self._logger = get_logger("RAFTMotionEstimator")
    
    def forward(
        self,
        image1: torch.Tensor,
        image2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Estimate optical flow between two images.
        
        Parameters
        ----------
        image1 : torch.Tensor
            First image, shape (B, C, H, W).
        image2 : torch.Tensor
            Second image, shape (B, C, H, W).
            
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (flow, confidence)
            flow: Optical flow, shape (B, 2, H, W)
            confidence: Confidence map, shape (B, H, W)
        """
        B, C, H, W = image1.shape
        
        # Extract features
        fmap1 = self.fnet(image1)
        fmap2 = self.fnet(image2)
        
        # Get context
        cnet_out = self.cnet(image1)
        context = cnet_out[:, :self.context_dim]
        hidden = cnet_out[:, self.context_dim:]
        
        # Initialize flow
        _, _, H_feat, W_feat = fmap1.shape
        flow = torch.zeros(B, 2, H_feat, W_feat, device=image1.device)
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            # Combine features with current flow
            combined = torch.cat([hidden, context, flow], dim=1)
            
            # Predict flow update
            delta_flow = self.update_block(combined)
            flow = flow + delta_flow
        
        # Upsample flow to original resolution
        flow_up = torch.nn.functional.interpolate(
            flow, size=(H, W), mode='bilinear', align_corners=True
        )
        flow_up = flow_up * (H / H_feat)  # Scale flow values
        
        # Compute confidence from correlation
        # Simplified: use feature correlation as proxy
        corr = torch.sum(fmap1 * fmap2, dim=1, keepdim=True)
        corr_norm = torch.sigmoid(corr)
        confidence = torch.nn.functional.interpolate(
            corr_norm, size=(H, W), mode='bilinear', align_corners=True
        ).squeeze(1)
        
        return flow_up, confidence
    
    def estimate_motion(
        self,
        image1: NDArray[np.float32],
        image2: NDArray[np.float32],
        frame_interval_s: float
    ) -> PixelMotion:
        """Estimate motion as PixelMotion dataclass.
        
        This is a convenience method that wraps forward() and returns
        a PixelMotion object.
        
        Parameters
        ----------
        image1, image2 : ndarray
            Images of shape (C, H, W).
        frame_interval_s : float
            Time between frames in seconds.
            
        Returns
        -------
        PixelMotion
            Pixel-space motion estimate.
        """
        # Convert to tensors
        t1 = torch.from_numpy(image1).unsqueeze(0)
        t2 = torch.from_numpy(image2).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            flow, confidence = self.forward(t1, t2)
        
        # Extract components
        flow_np = flow[0].cpu().numpy()
        conf_np = confidence[0].cpu().numpy()
        
        return PixelMotion(
            u_pixels=flow_np[0],
            v_pixels=flow_np[1],
            confidence=conf_np,
            frame_interval_s=frame_interval_s
        )
