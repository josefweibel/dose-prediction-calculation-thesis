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
      use_precomputed=True,
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
    s = 0.008  # Offset to prevent singularity at t=0
    steps = torch.linspace(0, 1, sampling_T, device=alphas_bar_train.device)

    # Cosine schedule: alpha_bar(t) = cos^2((t + s) / (1 + s) * pi/2)
    alphas_bar = torch.cos(((steps + s) / (1 + s)) * np.pi * 0.5) ** 2

    # Normalize to match training range (important for model compatibility)
    # Map [min_train, max_train] to our cosine curve
    min_alpha_train = alphas_bar_train.min()
    max_alpha_train = alphas_bar_train.max()

    # Rescale cosine to match training bounds
    alphas_bar = min_alpha_train + (max_alpha_train - min_alpha_train) * (
      (alphas_bar - alphas_bar.min()) / (alphas_bar.max() - alphas_bar.min() + 1e-8)
    )

    # Map to closest training timesteps for model conditioning
    distances = torch.abs(alphas_bar_train.unsqueeze(1) - alphas_bar.unsqueeze(0))
    indices = torch.argmin(distances, dim=0)
  elif schedule == "karras":
    # Log-spaced sigmas between high and low noise
    sigma_min, sigma_max, rho_val = 0.002, 80, 7.0
    i = torch.linspace(0, 1, sampling_T, device=alphas_bar_train.device)

    min_inv_rho = sigma_min ** (1 / rho_val)
    max_inv_rho = sigma_max ** (1 / rho_val)

    # Interpolate in rho-space and transform back
    sigmas = (max_inv_rho + i * (min_inv_rho - max_inv_rho)) ** rho_val
    # Add final sigma=0 for complete denoising
    sigmas = torch.cat([sigmas, torch.zeros(1, device=alphas_bar_train.device)])

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
    sampling_idx, alphas_bar_t = make_sampling_schedule(train_T, self.sampling_T, schedule, alphas_bar_train=)
    self.sampling_indices = sampling_idx.tolist()

    # For prev, shift by one position; last step goes to alpha_bar=1 (clean image)
    alphas_bar_prev = torch.zeros_like(alphas_bar_t)
    for i in range(len(alphas_bar_t)):
      if i < len(alphas_bar_t) - 1:
        alphas_bar_prev[i] = alphas_bar_t[i + 1]
      else:
        alphas_bar_prev[i] = 1.0  # Final step: clean image

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
      mean = self.mean_coef1[i] * x0_pred + self.mean_coef2[i] * x

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

    alphas_bar_prev = torch.zeros_like(alphas_bar_t)
    for i in range(len(sampling_idx)):
      if i < len(sampling_idx) - 1:
        alphas_bar_prev[i] = alphas_bar_t[sampling_idx[i + 1]]
      else:
        alphas_bar_prev[i] = 1.0  # Final step goes to clean image

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


def lagrange_weights(times, t_target):
  """
  times: 1D tensor/array of shape (m,) with distinct time points (floats)
  t_target: scalar float where we want to evaluate the interpolation polynomial
  returns: weights w_j such that p(t_target) = sum_j w_j * p(times[j])
  """
  times = torch.as_tensor(times, dtype=torch.float64)
  t = float(t_target)
  m = times.shape[0]
  w = torch.ones(m, dtype=torch.float64)
  for j in range(m):
    num = 1.0
    den = 1.0
    for r in range(m):
      if r == j:
        continue
      num *= (t - float(times[r]))
      den *= (float(times[j]) - float(times[r]))

    if abs(den) < 1e-12:
      # Degenerate case: duplicate times
      w[j] = 0.0
    else:
      w[j] = num / den

  return w.to(torch.get_default_dtype())  # cast back to default dtype

class DEISSampler(nn.Module):
  def __init__(
    self,
    model,
    beta_1,
    beta_T,
    train_T,
    cache_config,
    sampling_T=None,
    method: str = "rhoAB", # "rhoAB" or "tAB"
    order: int = 0, # 0 == DDIM-like; 1..k for multistep extrapolation
    schedule: str = "linear", # "linear", "cosine", or "karras"
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

    # Compute full training schedule
    betas = torch.linspace(beta_1, beta_T, self.train_T).float()
    alphas = 1. - betas
    alphas_bar_train = torch.cumprod(alphas, dim=0)

    # Get sampling indices
    sampling_idx, alphas_bar_t = make_sampling_schedule(train_T, self.sampling_T, schedule, alphas_bar_train)
    self.sampling_indices = sampling_idx.tolist()

    # For next step alpha_bar
    alphas_bar_prev = torch.zeros_like(alphas_bar_t)
    for i in range(len(sampling_idx)):
      if i < len(sampling_idx) - 1:
        alphas_bar_prev[i] = alphas_bar_t[sampling_idx[i + 1]]
      else:
        alphas_bar_prev[i] = 1.0

    sqrt_alphas_bar = torch.sqrt(alphas_bar_t)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar_t)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev, min=0.0))

    self.register_buffer("alphas_bar", alphas_bar_t)
    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("dir_coeff", dir_coeff)

    # Compute interpolation grid
    if self.method == "rhoAB" or schedule == "karras":
      grid = torch.log(alphas_bar_t / torch.clamp(1 - alphas_bar_t, min=1e-8))
    else:  # "tAB"
      idxs = torch.tensor(self.sampling_indices, dtype=torch.float32)
      grid = idxs / float(max(1, self.train_T - 1))

    # Precompute Lagrange weights for multistep extrapolation
    if use_precomputed and self.order > 0:
      precomp = []
      for i in range(len(grid)):
        if i + 1 >= len(grid):
          # No next step
          precomp.append(None)
          continue

        # Use up to (order+1) previous points
        max_m = min(self.order + 1, i + 1)
        prev_indices = [i - j for j in range(max_m)]
        prev_times = grid[prev_indices]
        t_target = float(grid[i + 1])

        weights = lagrange_weights(prev_times, t_target)
        precomp.append((prev_indices, weights))

      self._precomputed_weights = precomp
    else:
      self._precomputed_weights = None

  def forward(self, x_T, condition):
    x = x_T

    eps_buffer = deque(maxlen=self.order + 1)
    cache = DeepCache(self.cache_config)

    for i, t in enumerate(self.sampling_indices):
      cache.set_current_step(i)

      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      eps = self.model(x, condition, t_vec, cache)
      eps_buffer.append(eps)

      # Determine which epsilon to use for update
      if self.order == 0 or i + 1 >= len(self.sampling_indices):
        # Use current eps (DDIM-like)
        eps_for_update = eps
      else:
        # Extrapolate using Lagrange interpolation
        if self._precomputed_weights is not None and self._precomputed_weights[i] is not None:
          prev_indices, w = self._precomputed_weights[i]
          w = w.to(x.device)

          # Gather eps from buffer
          eps_list = []
          for rel in prev_indices:
            offset = i - rel
            idx_in_buffer = -(offset + 1)
            eps_list.append(eps_buffer[idx_in_buffer])

          # Apply weights
          stacked = torch.stack(eps_list, dim=0)
          w_shape = (w.shape[0],) + (1,) * (stacked.ndim - 1)
          w_reshaped = w.view(w_shape)
          eps_for_update = torch.sum(w_reshaped * stacked, dim=0)
        else:
          eps_for_update = eps

      # DDIM-style deterministic update
      sqrt_alpha_bar_t = self.sqrt_alphas_bar[i]
      sqrt_alpha_bar_prev = self.sqrt_alphas_bar_prev[i]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]
      dir_coeff = self.dir_coeff[i]

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps_for_update) / sqrt_alpha_bar_t
      x = sqrt_alpha_bar_prev * x0 + dir_coeff * eps_for_update

    return x
