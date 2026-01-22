"""
Warm-Start Diffusion (WSD) Implementation

Based on: "Warm-Start Diffusion: Accelerating Conditional Sampling"
by Scholz & Turner (2025)

Instead of starting from N(0, I), WSD:
1. Trains a warm-start model h_φ to predict N(μ_C, σ_C) from conditioning C
2. Normalizes targets during diffusion training: X'_t = (X_t - μ_C) / σ_norm
3. Uses warmth blending: σ_norm = w * max(σ_C, 1-w) + (1-w)

Two-phase training:
- Phase 1: Train h_φ with Gaussian NLL loss
- Phase 2: Train diffusion on normalized data using cached (μ, σ) predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from models import build_model


class WarmStartModel(nn.Module):
  """
  Wraps any U-Net to predict (μ, σ) for warm-start diffusion.

  The base model outputs 2 channels:
    - Channel 0: μ (mean prediction)
    - Channel 1: log_σ (log of standard deviation, transformed via softplus)

  Args:
    base_model: Any U-Net with out_channels=2
    min_sigma: Minimum sigma value for numerical stability (default: 0.01)
  """

  def __init__(self, base_model, min_sigma=0.01):
    super().__init__()
    self.base_model = base_model
    self.min_sigma = min_sigma

  def forward(self, condition):
    """
    Predict (μ, σ) from conditioning.

    Args:
      condition: Conditioning tensor [B, C, H, W]

    Returns:
      mu: Mean prediction [B, 1, H, W]
      sigma: Standard deviation prediction [B, 1, H, W]
    """
    out = self.base_model(condition)  # [B, 2, H, W]
    mu = out[:, 0:1]                  # [B, 1, H, W]
    log_sigma = out[:, 1:2]           # [B, 1, H, W]
    sigma = F.softplus(log_sigma) + self.min_sigma
    return mu, sigma


def build_warm_start_model(config, structure_selector):
  """
  Build warm-start model using any existing U-Net architecture.

  Uses the same model configuration as regular U-Net training, just with
  out_channels=2 to predict (μ, log_σ).

  Args:
    config: Model configuration dict with 'warm_start_model' key containing:
      - min_sigma: Minimum sigma value (default: 0.01)
      - Any model-specific config (deeplabv3plus, beamdiff, hdunet, c3d, mednext)
    structure_selector: Structure selector defining input channels

  Returns:
    WarmStartModel instance wrapping the specified architecture
  """

  ws_cfg = config['warm_start_model']
  min_sigma = ws_cfg.get('min_sigma', 0.01)

  # Build base model with out_channels=2 for (μ, log_σ) prediction
  base = build_model(config, structure_selector, out_channels=2)

  return WarmStartModel(base, min_sigma=min_sigma)


class WarmStartModelTrainerModule(nn.Module):
  """
  Trainer module for warm-start model (Phase 1).

  Uses Gaussian Negative Log-Likelihood loss:
    NLL = 0.5 * ((x_0 - μ) / σ)² + log(σ)

  Similar interface to UnetTrainerModule for compatibility with Trainer.
  """

  def __init__(self, model):
    super().__init__()
    self.model = model  # WarmStartModel

  def forward(self, x_0, condition):
    """
    Compute Gaussian NLL loss.

    Args:
      x_0: Target dose [B, 1, H, W]
      condition: Conditioning tensor [B, C, H, W]

    Returns:
      Scalar loss value
    """
    mu, sigma = self.model(condition)

    # Gaussian NLL: 0.5 * ((x_0 - mu) / sigma)^2 + log(sigma)
    normalized_diff = (x_0 - mu) / sigma
    nll = 0.5 * normalized_diff ** 2 + torch.log(sigma)

    return nll.mean()


class WarmStartDiffusionTrainerModule(nn.Module):
  """
  Wraps diffusion trainer to normalize targets using cached warm-start predictions (Phase 2).

  During training, samples warmth w ~ U[warmth_range] and normalizes:
    σ_norm = w * max(σ_C, 1-w) + (1-w)
    x_0_normalized = (x_0 - μ) / σ_norm
  """

  def __init__(self, base_trainer_module, warmth_range=(0.0, 1.0)):
    super().__init__()
    self.base_trainer = base_trainer_module
    self.warmth_range = warmth_range

  def forward(self, x_0, condition, mu, sigma):
    """
    Normalize target and delegate to base diffusion trainer.

    Args:
      x_0: Target dose [B, 1, H, W]
      condition: Conditioning tensor [B, C, H, W]
      mu: Cached warm-start mean [B, 1, H, W]
      sigma: Cached warm-start std [B, 1, H, W]

    Returns:
      Scalar loss value from base trainer
    """
    # Sample warmth w ~ U[warmth_range]
    w = torch.rand(x_0.shape[0], 1, 1, 1, device=x_0.device)
    w = self.warmth_range[0] + w * (self.warmth_range[1] - self.warmth_range[0])

    # Compute sigma_norm with warmth blending
    # σ_norm = w * max(σ_C, 1-w) + (1-w)
    sigma_norm = w * torch.maximum(sigma, 1.0 - w) + (1.0 - w)

    # Normalize target
    x_0_normalized = (x_0 - mu) / sigma_norm

    # Delegate to base diffusion trainer
    return self.base_trainer(x_0_normalized, condition)


class WarmStartUnetSampler(nn.Module):
  """
  Simple sampler for Phase 1 validation that returns μ without caching.

  Used during Phase 1 training to compute metrics comparing μ to ground truth.
  """

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x_T_ignored, condition):
    """
    Return μ prediction for metrics computation.

    Args:
      x_T_ignored: Ignored (for API compatibility)
      condition: Conditioning tensor [B, C, H, W]

    Returns:
      μ prediction [B, 1, H, W]
    """
    mu, _ = self.model(condition)
    return mu


class WarmStartCachingSampler(nn.Module):
  """
  'Sampler' that caches warm-start predictions to disk.

  Used after Phase 1 training to precompute (μ, σ) for Phase 2.
  Returns μ as the 'prediction' for metrics compatibility.
  """

  def __init__(self, model, cache_dir):
    super().__init__()
    self.model = model
    self.cache_dir = Path(cache_dir)
    self.cache_dir.mkdir(parents=True, exist_ok=True)

  def forward(self, x_T_ignored, condition, info=None):
    """
    Compute and cache (μ, σ) instead of sampling.

    Args:
      x_T_ignored: Ignored (for API compatibility)
      condition: Conditioning tensor [B, C, H, W]
      info: Batch info dict with 'volume' and 'index' keys

    Returns:
      μ prediction [B, 1, H, W] for metrics compatibility
    """
    mu, sigma = self.model(condition)

    # Save to disk using sample ID from info
    if info is not None:
      volumes = info.get('volume', [])
      indices = info.get('index', [])

      for i in range(mu.shape[0]):
        if i < len(volumes) and i < len(indices):
          volume = volumes[i]
          slice_idx = indices[i].item() if torch.is_tensor(indices[i]) else indices[i]
          sample_id = f"{volume}_{slice_idx}"

          torch.save({
            'mu': mu[i].cpu(),
            'sigma': sigma[i].cpu()
          }, self.cache_dir / f'{sample_id}.pt')

    return mu


class WarmStartCachedDataset(Dataset):
  """
  Wraps base dataset to include cached (μ, σ) predictions for Phase 2 training.
  """

  def __init__(self, base_dataset, cache_dir):
    self.base_dataset = base_dataset
    self.cache_dir = Path(cache_dir)

    # Verify cache directory exists
    if not self.cache_dir.exists():
      raise FileNotFoundError(
        f"Warm-start cache not found at {self.cache_dir}. "
        f"Run Phase 1 training first with warm_start_cache sampler."
      )

  def __len__(self):
    return len(self.base_dataset)

  def __getitem__(self, idx):
    """
    Get item with cached warm-start predictions.

    Returns:
      condition: Conditioning tensor [C, H, W]
      dose: Target dose [1, H, W]
      mu: Cached mean prediction [1, H, W]
      sigma: Cached std prediction [1, H, W]
      info: Original info dict
    """
    condition, dose, info = self.base_dataset[idx]

    # Build sample ID matching the caching format
    volume = info['volume']
    slice_idx = info['index']
    sample_id = f"{volume}_{slice_idx}"

    cache_path = self.cache_dir / f'{sample_id}.pt'
    if not cache_path.exists():
      raise FileNotFoundError(
        f"Cached warm-start prediction not found: {cache_path}. "
        f"Ensure Phase 1 caching was run for all samples."
      )

    cached = torch.load(cache_path, weights_only=True)

    return condition, dose, cached['mu'], cached['sigma'], info

  @property
  def selector(self):
    """Forward selector access to base dataset."""
    return self.base_dataset.selector

  @property
  def dose_range(self):
    """Forward dose_range access to base dataset."""
    return self.base_dataset.dose_range

  def denormalize_target(self, target):
    """Forward denormalization to base dataset."""
    return self.base_dataset.denormalize_target(target)

  def finish(self):
    """Forward finish to base dataset if it exists."""
    if hasattr(self.base_dataset, 'finish'):
      self.base_dataset.finish()


class WarmStartSampler(nn.Module):
  """
  Wraps any existing sampler for warm-start inference.

  At inference time:
  1. Uses provided (μ, σ) or computes from warm-start model
  2. Runs base sampler in normalized space
  3. Unnormalizes the result

  Args:
    base_sampler: Any existing sampler (DDPM, DDIM, DEIS, etc.)
    warm_start_model: Trained WarmStartModel (optional if using cached values)
    warmth: Fixed warmth value for inference (default: 1.0 = full warm-start)
  """

  def __init__(self, base_sampler, warm_start_model=None, warmth=1.0):
    super().__init__()
    self.base_sampler = base_sampler
    self.warm_start_model = warm_start_model
    self.warmth = warmth

  def forward(self, x_T, condition, mu=None, sigma=None):
    """
    Sample with warm-start.

    Args:
      x_T: Initial noise [B, 1, H, W]
      condition: Conditioning tensor [B, C, H, W]
      mu: Pre-computed mean [B, 1, H, W] (optional, computed if not provided)
      sigma: Pre-computed std [B, 1, H, W] (optional, computed if not provided)

    Returns:
      Denoised sample [B, 1, H, W]
    """
    # Use provided (mu, sigma) or compute from model
    if mu is None or sigma is None:
      if self.warm_start_model is None:
        raise ValueError("Either provide (mu, sigma) or warm_start_model")
      with torch.no_grad():
        mu, sigma = self.warm_start_model(condition)

    # Compute sigma_norm with fixed warmth
    w = self.warmth
    sigma_norm = w * torch.maximum(sigma, torch.tensor(1.0 - w, device=sigma.device)) + (1.0 - w)

    # Run base sampler in normalized space
    x_0_normalized = self.base_sampler(x_T, condition)

    # Unnormalize
    x_0 = x_0_normalized * sigma_norm + mu

    return x_0


def compute_sigma_norm(sigma, warmth):
  """
  Compute normalized sigma with warmth blending.

  σ_norm = w * max(σ_C, 1-w) + (1-w)

  Args:
    sigma: Predicted standard deviation [B, 1, H, W] or scalar
    warmth: Warmth value (0 = cold start from N(0,1), 1 = full warm-start)

  Returns:
    sigma_norm with same shape as sigma
  """
  w = warmth
  if torch.is_tensor(sigma):
    return w * torch.maximum(sigma, torch.tensor(1.0 - w, device=sigma.device)) + (1.0 - w)
  else:
    return w * max(sigma, 1.0 - w) + (1.0 - w)


def normalize_target(x_0, mu, sigma_norm):
  """
  Normalize target for training.

  X'_0 = (X_0 - μ) / σ_norm

  Args:
    x_0: Clean sample [B, 1, H, W]
    mu: Mean prediction [B, 1, H, W]
    sigma_norm: Normalized sigma [B, 1, H, W]

  Returns:
    Normalized sample [B, 1, H, W]
  """
  return (x_0 - mu) / sigma_norm


def unnormalize_sample(x_0_norm, mu, sigma_norm):
  """
  Unnormalize sample after diffusion.

  X_0 = X'_0 * σ_norm + μ

  Args:
    x_0_norm: Normalized sample [B, 1, H, W]
    mu: Mean prediction [B, 1, H, W]
    sigma_norm: Normalized sigma [B, 1, H, W]

  Returns:
    Unnormalized sample [B, 1, H, W]
  """
  return x_0_norm * sigma_norm + mu
