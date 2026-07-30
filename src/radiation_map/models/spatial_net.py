"""Deep Spatial Neural Regressor with Spherical Harmonic Fourier Encodings & Heteroscedastic Uncertainty."""

import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from radiation_map.data.transforms import (
    latlon_to_cartesian,
    inverse_log1p_target_transform,
)


class MultiScaleFourierEmbedding(nn.Module):
    """Encodes 3D Cartesian coordinates on the unit sphere into multi-scale Fourier features."""
    
    def __init__(self, in_dim: int = 3, num_frequencies: int = 32, max_frequency_log: float = 4.0):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        
        # Log-spaced frequency bands from 2^0 to 2^max_frequency_log
        freq_bands = 2.0 ** torch.linspace(0.0, max_frequency_log, num_frequencies)
        self.register_buffer("freq_bands", freq_bands)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Args:
            x: Tensor of shape (batch_size, 3) representing (x, y, z) on unit sphere
        Returns:
            Tensor of shape (batch_size, 3 + 2 * 3 * num_frequencies)
        """
        # (batch_size, 3, 1) * (1, 1, num_frequencies) -> (batch_size, 3, num_frequencies)
        scaled = x.unsqueeze(-1) * self.freq_bands.view(1, 1, -1) * math.pi
        
        sin_features = torch.sin(scaled).reshape(x.shape[0], -1)
        cos_features = torch.cos(scaled).reshape(x.shape[0], -1)
        
        return torch.cat([x, sin_features, cos_features], dim=-1)
        
    @property
    def output_dim(self) -> int:
        return self.in_dim + 2 * self.in_dim * self.num_frequencies


class ResidualBlock(nn.Module):
    """Residual MLP Block with LayerNorm, SiLU activation, and Dropout."""
    
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.act(self.norm1(self.fc1(x)))
        out = self.dropout(out)
        out = self.norm2(self.fc2(out))
        out = self.act(out + residual)
        return out


class SpatialHarmonicNet(nn.Module):
    """Spatial Neural Network predicting expected radiation level and aleatoric uncertainty.
    
    Input:
        (latitude, longitude) or 3D Cartesian coordinates (x, y, z)
    Output:
        mu: Predicted mean radiation in log space log(1 + 10 * y)
        log_var: Predicted aleatoric uncertainty log(sigma^2)
    """
    
    def __init__(
        self,
        in_dim: int = 3,
        num_frequencies: int = 32,
        max_frequency_log: float = 4.5,
        hidden_dims: Tuple[int, ...] = (256, 256, 128, 64),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.num_frequencies = num_frequencies
        self.max_frequency_log = max_frequency_log
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Spatial multi-scale embedding
        self.embedding = MultiScaleFourierEmbedding(
            in_dim=in_dim,
            num_frequencies=num_frequencies,
            max_frequency_log=max_frequency_log,
        )
        embed_dim = self.embedding.output_dim
        
        # Input projection
        self.input_layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.SiLU(),
            nn.Dropout(p=dropout),
        )
        
        # Backbone residual blocks
        blocks = []
        for i in range(len(hidden_dims) - 1):
            blocks.append(ResidualBlock(hidden_dims[i], dropout=dropout))
            if hidden_dims[i] != hidden_dims[i + 1]:
                blocks.append(nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]),
                    nn.LayerNorm(hidden_dims[i + 1]),
                    nn.SiLU(),
                    nn.Dropout(p=dropout),
                ))
        self.backbone = nn.Sequential(*blocks)
        
        last_dim = hidden_dims[-1]
        
        # Dual Heads: Mean and Log-Variance
        self.head_mu = nn.Sequential(
            nn.Linear(last_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
        self.head_logvar = nn.Sequential(
            nn.Linear(last_dim, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )
        
    def forward(self, coords_cartesian: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass predicting (mu, log_var) in log-transformed space."""
        # Multi-scale positional encoding
        embedded = self.embedding(coords_cartesian)
        
        # Feature extraction
        h = self.input_layer(embedded)
        h = self.backbone(h)
        
        mu = self.head_mu(h)
        log_var = self.head_logvar(h)
        
        # Clip log_var to reasonable dynamic range [-7, 3] for numerical stability
        log_var = torch.clamp(log_var, min=-7.0, max=3.0)
        return mu, log_var
        
    @torch.no_grad()
    def predict_radiation(
        self,
        latitudes: Union[float, np.ndarray, torch.Tensor],
        longitudes: Union[float, np.ndarray, torch.Tensor],
        num_mc_samples: int = 1,
        device: Optional[torch.device] = None,
    ) -> Dict[str, np.ndarray]:
        """Inference method returning predictions in physical uSv/h scale with uncertainty."""
        if device is None:
            device = next(self.parameters()).device
            
        self.eval()
        
        # Convert lat/lon to Cartesian tensor
        if not isinstance(latitudes, torch.Tensor):
            xyz = latlon_to_cartesian(latitudes, longitudes)
            if xyz.ndim == 1:
                xyz = xyz.reshape(1, 3)
            coords_tensor = torch.tensor(xyz, dtype=torch.float32, device=device)
        else:
            coords_tensor = latitudes.to(device)
            
        if num_mc_samples > 1:
            # Enable dropout for Monte Carlo epistemic uncertainty estimation
            self.train()
            mu_samples = []
            var_samples = []
            for _ in range(num_mc_samples):
                mu_s, logvar_s = self(coords_tensor)
                mu_samples.append(mu_s.unsqueeze(0))
                var_samples.append(torch.exp(logvar_s).unsqueeze(0))
                
            mu_stack = torch.cat(mu_samples, dim=0) # (samples, batch, 1)
            var_stack = torch.cat(var_samples, dim=0)
            
            # Mean of predictive distributions
            mu_log = mu_stack.mean(dim=0).squeeze(-1).cpu().numpy()
            aleatoric_var = var_stack.mean(dim=0).squeeze(-1).cpu().numpy()
            epistemic_var = mu_stack.var(dim=0).squeeze(-1).cpu().numpy()
            total_var = aleatoric_var + epistemic_var
            sigma_log = np.sqrt(np.maximum(total_var, 1e-8))
            self.eval()
        else:
            mu_tensor, logvar_tensor = self(coords_tensor)
            mu_log = mu_tensor.squeeze(-1).cpu().numpy()
            sigma_log = torch.exp(0.5 * logvar_tensor).squeeze(-1).cpu().numpy()
            aleatoric_var = (sigma_log ** 2)
            epistemic_var = np.zeros_like(aleatoric_var)
            
        # Convert from log space back to physical uSv/h scale
        radiation_usv = inverse_log1p_target_transform(mu_log)
        
        # Confidence lower and upper bounds (95% approx +/- 1.96 sigma in log space)
        lower_log = np.maximum(mu_log - 1.96 * sigma_log, 0.0)
        upper_log = mu_log + 1.96 * sigma_log
        
        radiation_usv_low = inverse_log1p_target_transform(lower_log)
        radiation_usv_high = inverse_log1p_target_transform(upper_log)
        
        # Physical standard deviation approximation via delta method
        derivative = np.exp(mu_log) / 10.0
        sigma_physical = sigma_log * derivative
        
        # Squeeze if scalar
        is_scalar = np.isscalar(latitudes) and np.isscalar(longitudes)
        if is_scalar:
            return {
                "radiation_usv": float(np.asarray(radiation_usv).item()),
                "uncertainty_usv": float(np.asarray(sigma_physical).item()),
                "ci_lower_usv": float(np.asarray(radiation_usv_low).item()),
                "ci_upper_usv": float(np.asarray(radiation_usv_high).item()),
                "mu_log": float(np.asarray(mu_log).item()),
                "sigma_log": float(np.asarray(sigma_log).item()),
                "aleatoric_var": float(np.asarray(aleatoric_var).item()),
                "epistemic_var": float(np.asarray(epistemic_var).item()),
            }
        
        return {
            "radiation_usv": np.asarray(radiation_usv).flatten(),
            "uncertainty_usv": np.asarray(sigma_physical).flatten(),
            "ci_lower_usv": np.asarray(radiation_usv_low).flatten(),
            "ci_upper_usv": np.asarray(radiation_usv_high).flatten(),
            "mu_log": np.asarray(mu_log).flatten(),
            "sigma_log": np.asarray(sigma_log).flatten(),
            "aleatoric_var": np.asarray(aleatoric_var).flatten(),
            "epistemic_var": np.asarray(epistemic_var).flatten(),
        }
