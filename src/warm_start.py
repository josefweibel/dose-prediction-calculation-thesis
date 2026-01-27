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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from pathlib import Path

from models import build_model
from config import read_model_config


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
  Sampler for warm-start model that returns (μ, σ) predictions.

  Used for Phase 1 validation and caching. Returns dict with 'pred' (μ)
  and 'sigma' for evaluator to save.
  """

  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x_T_ignored, condition):
    """
    Compute (μ, σ) and return as dict.

    Args:
      x_T_ignored: Ignored (for API compatibility)
      condition: Conditioning tensor [B, C, H, W]

    Returns:
      dict with 'pred' (μ) and 'sigma' tensors [B, 1, H, W]
    """
    mu, sigma = self.model(condition)

    return {
      'pred': mu,
      'sigma': sigma
    }


class WarmStartCachedDataset(Dataset):
  """
  Wraps base dataset to include cached (μ, σ) predictions for Phase 2 training.

  Expects combined volume files from SliceCombinator:
    - {volume}.npy for pred/mu (shape: [C, H, W, num_slices])
    - {volume}_sigma.npy for sigma (shape: [C, H, W, num_slices])
  """

  def __init__(self, base_dataset, ws_model_name, split, cache_volumes=True):
    self.base_dataset = base_dataset
    self.cache_volumes = cache_volumes
    self._volume_cache = {} if cache_volumes else None

    # Resolve cache directory from warm-start model config
    ws_model_config = read_model_config(ws_model_name)
    ws_epoch = ws_model_config['epochs'] - 1
    self.cache_dir = Path('samples') / ws_model_name / 'warm-start-unet' / f'{split}_{ws_epoch}'

    # Verify cache directory exists
    if not self.cache_dir.exists():
      raise FileNotFoundError(
        f"Warm-start cache not found at {self.cache_dir}. "
        f"Run Phase 1 caching and SliceCombinator first."
      )

  def _load_volume(self, volume_name):
    """Load volume data, with optional caching."""
    if self.cache_volumes and volume_name in self._volume_cache:
      return self._volume_cache[volume_name]

    mu_path = self.cache_dir / f'{volume_name}.npy'
    sigma_path = self.cache_dir / f'{volume_name}_sigma.npy'

    if not mu_path.exists():
      raise FileNotFoundError(
        f"Cached mu volume not found: {mu_path}. "
        f"Ensure Phase 1 caching and SliceCombinator were run."
      )
    if not sigma_path.exists():
      raise FileNotFoundError(
        f"Cached sigma volume not found: {sigma_path}. "
        f"Ensure Phase 1 caching and SliceCombinator were run."
      )

    if self.cache_volumes:
      mu_vol = np.load(mu_path)
      sigma_vol = np.load(sigma_path)
      self._volume_cache[volume_name] = (mu_vol, sigma_vol)
    else:
      # Use memmap to avoid loading entire volume
      mu_vol = np.load(mu_path, mmap_mode='r')
      sigma_vol = np.load(sigma_path, mmap_mode='r')

    return mu_vol, sigma_vol

  def preload_cache(self):
    """
    Pre-populate the volume cache in the main process.

    Call this BEFORE creating DataLoader workers so that forked workers
    inherit the populated cache via copy-on-write, avoiding duplicate loading.
    """
    if not self.cache_volumes:
      return

    # Get unique volume names from base dataset's index
    volume_names = self.base_dataset.index['volume'].unique()

    # Pre-load all volumes
    for volume_name in volume_names:
      self._load_volume(volume_name)

  @property
  def index(self):
    """Delegate index access to base dataset for compatibility."""
    return self.base_dataset.index

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

    volume_name = info['volume']
    slice_idx = info['index']

    mu_vol, sigma_vol = self._load_volume(volume_name)

    mu = torch.from_numpy(mu_vol[..., slice_idx].copy())
    sigma = torch.from_numpy(sigma_vol[..., slice_idx].copy())

    # Normalize mu from Gy range to [0, 1] range
    # (evaluator saves denormalized predictions, but training expects normalized values)
    mu = self.base_dataset.normalize_target(mu)

    return condition, dose, mu, sigma, info

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


class InDIWarmStartSampler(nn.Module):
  """
  Wrapper for InDI sampler that uses warm-start μ as degraded input.

  Unlike WarmStartSampler which normalizes/unnormalizes, this wrapper
  simply passes μ directly to the InDI sampler as the degraded input y.

  Args:
    base_sampler: InDISampler instance
    warm_start_model: Trained WarmStartModel (optional if using cached values)
  """

  def __init__(self, base_sampler, warm_start_model=None):
    super().__init__()
    self.base_sampler = base_sampler
    self.warm_start_model = warm_start_model

  def forward(self, x_T_ignored, condition, mu=None, sigma=None):
    """
    Sample using InDI with warm-start μ as degraded input.

    Args:
      x_T_ignored: Ignored (for API compatibility with other samplers)
      condition: Conditioning tensor [B, C, H, W]
      mu: Pre-computed mean [B, 1, H, W] (degraded input y for InDI)
      sigma: Pre-computed std [B, 1, H, W] (currently unused by InDI)

    Returns:
      Generated clean sample [B, 1, H, W]
    """
    # Use provided mu or compute from model
    if mu is None:
      if self.warm_start_model is None:
        raise ValueError("Either provide mu or warm_start_model")
      with torch.no_grad():
        mu, sigma = self.warm_start_model(condition)

    # InDI uses μ directly as the degraded input y
    return self.base_sampler(mu, condition)


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
