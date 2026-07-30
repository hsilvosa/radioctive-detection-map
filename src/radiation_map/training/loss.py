"""Loss functions for heteroscedastic spatial regression."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class HeteroscedasticGaussianLoss(nn.Module):
    """Gaussian Negative Log Likelihood (NLL) Loss with variance regularization.
    
    L(mu, log_var, y) = 0.5 * (log_var + (y - mu)^2 / exp(log_var)) + 0.5 * log(2 * pi)
    """
    
    def __init__(self, eps: float = 1e-6, huber_delta: float = 3.0):
        super().__init__()
        self.eps = eps
        self.huber_delta = huber_delta
        
    def forward(self, mu: torch.Tensor, log_var: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Args:
            mu: Predicted mean (batch_size, 1)
            log_var: Predicted log variance (batch_size, 1)
            target: Ground truth target (batch_size, 1)
        """
        diff = target - mu
        var = torch.exp(log_var) + self.eps
        
        # Robust NLL calculation: using Huber threshold on normalized squared residuals
        sq_err = diff ** 2
        
        # Gaussian NLL
        nll = 0.5 * (log_var + sq_err / var)
        return torch.mean(nll)
