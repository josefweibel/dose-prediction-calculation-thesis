import torch
from torch import nn
from torch.nn import functional as F
from collections import deque
import numpy as np

from model_cache import DeepCache, CacheConfig


def build_sampler(model_config, sampling_config, model):
  if 'cache' in sampling_config:
    intervals = {
      key: np.inf if val == 'inf' else val
      for key, val in sampling_config['cache']['intervals'].items()
    }

    cache_config = CacheConfig(
      intervals=intervals,
      warmup=sampling_config['cache']['warmup_steps'],
    )
  else:
    cache_config = CacheConfig(intervals={}, warmup=0)

  if sampling_config['sampler'] == 'ddpm':
    return GaussianDiffusionSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      cache_config,
      sampling_config['T'],
      schedule=sampling_config['schedule'],
    )
  elif sampling_config['sampler'] == 'ddim':
    return DDIMSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      cache_config,
      sampling_config['T'],
      eta=sampling_config['eta'],
      schedule=sampling_config['schedule'],
    )
  elif sampling_config['sampler'] == 'deis':
    return DEISSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      cache_config,
      sampling_config['T'],
      method=sampling_config['method'],
      order=sampling_config['order'],
      schedule=sampling_config['schedule'],
    )
  else:
    raise ValueError(f"Unknown sampler: {sampling_config['sampler']}")


def make_sampling_schedule(train_T, sampling_T, schedule, alphas_bar_train):
  """
  Create sampling indices for different time schedules.
  Args:
      train_T: total training steps (e.g. 1000)
      sampling_T: number of inference steps (e.g. 50)
      schedule: 'linear', 'cosine', or 'karras'
  Returns:
      sampling_indices (torch.LongTensor)
      alphas_bar to use for sampling (torch.Tensor)
  """
  if schedule == "linear":
    # Uniform in t
    indices = torch.linspace(train_T - 1, 0, sampling_T, dtype=torch.long)
    alphas_bar = alphas_bar_train[indices]
  elif schedule == "cosine":
    eps = 1e-8
    rho_train = torch.log(alphas_bar_train / (1 - alphas_bar_train + eps))

    # Sample evenly in log-SNR space (from high noise to low noise)
    # rho_train[0] is at t=0 (low noise), rho_train[-1] is at t=T-1 (high noise)
    rho_target = torch.linspace(rho_train[-1], rho_train[0], sampling_T)

    # Map to closest training timesteps
    distances = torch.abs(rho_train.unsqueeze(1) - rho_target.unsqueeze(0))
    indices = torch.argmin(distances, dim=0)

    alphas_bar = alphas_bar_train[indices]
  elif schedule == "karras":
    # Log-spaced sigmas between high and low noise
    sigma_min, sigma_max, rho_val = 0.002, 80, 7.0
    i = torch.linspace(0, 1, sampling_T, device=alphas_bar_train.device)

    min_inv_rho = sigma_min ** (1 / rho_val)
    max_inv_rho = sigma_max ** (1 / rho_val)

    # Interpolate in rho-space and transform back
    sigmas = (max_inv_rho + i * (min_inv_rho - max_inv_rho)) ** rho_val
    # Add final sigma=0 for complete denoising
    sigmas[-1] = 0.0

    # convert to corresponding alpha_bar values
    alphas_bar = 1 / (1 + sigmas**2)

    distances = torch.abs(alphas_bar_train.unsqueeze(1) - alphas_bar.unsqueeze(0))
    indices = torch.argmin(distances, dim=0)
  else:
    raise ValueError(f"Unknown schedule: {schedule}")

  print(f"Sampling indices (n={len(indices)}): {indices}")
  return indices, alphas_bar

class GaussianDiffusionSampler(nn.Module):
  def __init__(
    self,
    model,
    beta_1,
    beta_T,
    train_T,
    cache_config,
    sampling_T=None,
    schedule: str = "linear", # "linear", "cosine", or "karras"
  ):
    super().__init__()

    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T
    self.cache_config = cache_config

    # Compute full training schedule
    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1. - betas
    alphas_bar_train = torch.cumprod(alphas, dim=0)

    # Get sampling indices
    sampling_idx, alphas_bar_t = make_sampling_schedule(train_T, self.sampling_T, schedule, alphas_bar_train)
    self.sampling_indices = sampling_idx.tolist()

    alphas_bar_prev = torch.cat([alphas_bar_t[1:], torch.ones(1, device=alphas_bar_t.device)])

    # Compute effective betas for each transition
    betas_eff = 1.0 - (alphas_bar_t / torch.clamp(alphas_bar_prev, min=1e-8))
    betas_eff = torch.clamp(betas_eff, min=0.0, max=0.999)

    # Posterior variance for DDPM
    posterior_variance = betas_eff * (1 - alphas_bar_prev) / torch.clamp(1 - alphas_bar_t, min=1e-8)
    posterior_variance = torch.clamp(posterior_variance, min=0.0)

    # Cache only what's needed in forward pass
    self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar_t))
    self.register_buffer("sqrt_recip_alphas_bar", torch.sqrt(1. / torch.clamp(alphas_bar_t, min=1e-8)))
    self.register_buffer("posterior_std", torch.sqrt(posterior_variance))

    # Precompute mean calculation coefficients
    sqrt_alpha_bar_prev = torch.sqrt(alphas_bar_prev)
    sqrt_one_minus_betas_eff = torch.sqrt(1 - betas_eff)
    one_minus_alphas_bar_t_recip = 1.0 / torch.clamp(1 - alphas_bar_t, min=1e-8)

    # Cache mean coefficients: coef1 * x0_pred + coef2 * x_t
    self.register_buffer("mean_coef1", sqrt_alpha_bar_prev * betas_eff * one_minus_alphas_bar_t_recip)
    self.register_buffer("mean_coef2", sqrt_one_minus_betas_eff * (1 - alphas_bar_prev) * one_minus_alphas_bar_t_recip)

  def forward(self, x_T, condition):
    x = x_T

    cache = DeepCache(self.cache_config)
    for i, t in enumerate(self.sampling_indices):
      cache.set_current_step(i)

      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      eps = self.model(x, condition, t_vec, cache)

      # Predict x0 using cached coefficients
      x0_pred = self.sqrt_recip_alphas_bar[i] * (x - self.sqrt_one_minus_alphas_bar[i] * eps)

      # Compute mean using precomputed coefficients
      mean = self.mean_coef1[i] * x0_pred + self.mean_coef2[i] * x # Eq. 7 in DDPM paper

      x = mean

      # Add noise (not on last step)
      if i < len(self.sampling_indices) - 1:
        noise = torch.randn_like(x)
        x += self.posterior_std[i] * noise

    return x

class DDIMSampler(nn.Module):
  def __init__(
    self,
    model,
    beta_1,
    beta_T,
    train_T,
    cache_config,
    sampling_T=None,
    eta=0.0,
    schedule: str = "linear", # "linear", "cosine", or "karras"
  ):
    super().__init__()
    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T
    self.cache_config = cache_config
    self.eta = eta

    # Compute full training schedule
    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1. - betas
    alphas_bar_train = torch.cumprod(alphas, 0)

    # Get sampling indices
    sampling_idx, alphas_bar_t = make_sampling_schedule(train_T, self.sampling_T, schedule, alphas_bar_train)
    self.sampling_indices = sampling_idx.tolist()

    alphas_bar_prev = torch.cat([alphas_bar_t[1:], torch.ones(1, device=alphas_bar_t.device)])

    sqrt_alphas_bar = torch.sqrt(alphas_bar_t)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar_t)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)

    sigma = torch.zeros_like(alphas_bar_t)
    if eta > 0:
      frac = (1 - alphas_bar_prev) / torch.clamp(1 - alphas_bar_t, min=1e-8)
      inner = 1 - (alphas_bar_t / torch.clamp(alphas_bar_prev, min=1e-8))
      sigma = eta * torch.sqrt(torch.clamp(frac * inner, min=0.0))

    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev - sigma**2, min=0.0))

    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("sigma", sigma)
    self.register_buffer("dir_coeff", dir_coeff)

  def forward(self, x_T, condition):
    x = x_T

    cache = DeepCache(self.cache_config)
    for i, t in enumerate(self.sampling_indices):
      cache.set_current_step(i)

      t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
      eps = self.model(x, condition, t_vec, cache)

      sqrt_alpha_bar_t = self.sqrt_alphas_bar[i]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]
      sqrt_alpha_bar_prev = self.sqrt_alphas_bar_prev[i]
      sigma = self.sigma[i]
      dir_coeff = self.dir_coeff[i]

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
      x = sqrt_alpha_bar_prev * x0 + dir_coeff * eps

      if i < len(self.sampling_indices) - 1 and sigma > 0:
        noise = torch.randn_like(x)
        x += sigma * noise

    return x

def lagrange_weights(x_points: torch.Tensor, x_target: float) -> torch.Tensor:
  """Compute Lagrange interpolation weights for extrapolation."""
  n = len(x_points)
  weights = torch.zeros(n)

  for j in range(n):
    numerator = 1.0
    denominator = 1.0
    for m in range(n):
      if m != j:
        numerator *= (x_target - float(x_points[m]))
        denominator *= (float(x_points[j]) - float(x_points[m]))
    weights[j] = numerator / denominator

  return weights

class DEISSampler(nn.Module):
  """
  DEIS (Diffusion Exponential Integrator Sampler) using DDIM-style updates
  with polynomial extrapolation of epsilon for higher-order accuracy.
  """

  def __init__(
    self,
    model,
    beta_1: float,
    beta_T: float,
    train_T: int,
    cache_config,
    sampling_T: int = None,
    method: str = "rhoAB",
    order: int = 0,  # 0 = DDIM, 1+ = multistep DEIS
    schedule: str = "linear",
    use_precomputed: bool = True,
  ):
    super().__init__()
    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T
    self.cache_config = cache_config
    self.method = method
    assert method in ("rhoAB", "tAB"), "method must be 'rhoAB' or 'tAB'"
    self.order = order
    assert self.order >= 0, "order must be >= 0"

    # Compute noise schedule
    betas = torch.linspace(beta_1, beta_T, self.train_T).float()
    alphas = 1.0 - betas
    alphas_bar_train = torch.cumprod(alphas, dim=0)

    # Get sampling schedule (decreasing order for denoising)
    sampling_idx, alphas_bar_t = make_sampling_schedule(
      train_T, self.sampling_T, schedule, alphas_bar_train
    )
    self.sampling_indices = sampling_idx.tolist()

    # Precompute DDIM coefficients
    self.register_buffer("alphas_bar", alphas_bar_t)

    # For next step alpha_bar (shift by one)
    alphas_bar_prev = torch.cat([alphas_bar_t[1:], torch.zeros(1, device=alphas_bar_t.device)])

    sqrt_alphas_bar = torch.sqrt(alphas_bar_t)
    sqrt_one_minus_alphas_bar = torch.sqrt(1.0 - alphas_bar_t)
    sqrt_alphas_bar_prev = torch.sqrt(torch.clamp(alphas_bar_prev, min=0.0))
    dir_coeff = torch.sqrt(torch.clamp(1.0 - alphas_bar_prev, min=0.0))

    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("dir_coeff", dir_coeff)

    # Compute integration grid for extrapolation
    if self.method == "rhoAB":
      grid = torch.log(alphas_bar_t / torch.clamp(1.0 - alphas_bar_t, min=1e-8))
    else:  # tAB
      grid = sampling_idx / float(max(1, self.train_T - 1))

    self.register_buffer("grid", grid)

    # Precompute Lagrange weights for extrapolation
    if use_precomputed and self.order > 0:
      precomp = []
      for i in range(len(grid)):
        if i + 1 >= len(grid):
          precomp.append(None)
          continue

        # Use up to (order+1) points: current i and previous i-1, i-2, ...
        max_history = min(self.order + 1, i + 1)
        history_indices = list(range(i, i - max_history, -1))
        history_grid = grid[history_indices]
        target_grid = float(grid[i + 1])

        weights = lagrange_weights(history_grid, target_grid)
        precomp.append((history_indices, weights))

      self._precomputed_weights = precomp
    else:
      self._precomputed_weights = None

  def forward(self, x_T, condition):
    x = x_T

    # Buffer for storing recent epsilon predictions
    eps_buffer = deque(maxlen=self.order + 1 if self.order > 0 else 1)
    cache = DeepCache(self.cache_config)

    for i in range(len(self.sampling_indices)):
      t = self.sampling_indices[i]
      cache.set_current_step(i)

      # Predict noise
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      eps = self.model(x, condition, t_vec, cache)
      eps_buffer.append(eps)

      # Last step - no update needed
      if i + 1 >= len(self.sampling_indices):
        break

      # Determine epsilon to use for update
      if self.order == 0:
        # Order 0: DDIM, use current epsilon
        eps_for_update = eps
      else:
        # Higher order: extrapolate using polynomial
        if self._precomputed_weights is not None and self._precomputed_weights[i] is not None:
          history_indices, weights = self._precomputed_weights[i]
          weights = weights.to(x.device)

          # Gather epsilon predictions
          eps_list = []
          for hist_idx in history_indices:
            buffer_idx = hist_idx - i  # 0, -1, -2, ... (relative to current i)
            eps_list.append(eps_buffer[buffer_idx])

          # Apply Lagrange weights
          stacked = torch.stack(eps_list, dim=0)
          w_shape = (weights.shape[0],) + (1,) * (stacked.ndim - 1)
          w_reshaped = weights.view(w_shape)
          eps_for_update = torch.sum(w_reshaped * stacked, dim=0)
        else:
          eps_for_update = eps

      # DDIM update: predict x0, then compute x_{t-1}
      sqrt_alpha_bar = self.sqrt_alphas_bar[i]
      sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_bar[i]
      sqrt_alpha_bar_prev = self.sqrt_alphas_bar_prev[i]
      dir_coeff = self.dir_coeff[i]

      # Predict x0 from current x_t and eps
      x0_pred = (x - sqrt_one_minus_alpha_bar * eps_for_update) / sqrt_alpha_bar

      # Compute x_{t-1} (DDIM deterministic update)
      x = sqrt_alpha_bar_prev * x0_pred + dir_coeff * eps_for_update

    return x
