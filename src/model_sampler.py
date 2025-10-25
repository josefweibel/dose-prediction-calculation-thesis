import torch
from torch import nn
from torch.nn import functional as F
import math
from collections import deque
import numpy as np


def build_sampler(model_config, sampling_config, model):
  if sampling_config['sampler'] == 'ddpm':
    return GaussianDiffusionSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      sampling_config['T'],
      schedule=sampling_config['schedule'],
    )
  elif sampling_config['sampler'] == 'ddim':
    return DDIMSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
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
      sampling_config['T'],
      method=sampling_config['method'],
      order=sampling_config['order'],
      schedule=sampling_config['schedule'],
      use_precomputed=True,
    )
  else:
    raise ValueError(f"Unknown sampler: {sampling_config['sampler']}")


def make_sampling_indices(train_T, sampling_T, schedule, alphas_bar):
  """
  Create sampling indices for different time schedules.
  Args:
      train_T: total training steps (e.g. 1000)
      sampling_T: number of inference steps (e.g. 50)
      schedule: 'linear', 'cosine', or 'karras'
  Returns:
      sampling_indices (torch.LongTensor)
  """
  if schedule == "linear":
    # Uniform in t
    indices = torch.linspace(train_T - 1, 0, sampling_T, dtype=torch.long)

  elif schedule == "cosine":
    # Map cosine cumulative alphas to training timesteps
    s = 0.008
    steps = torch.linspace(0, 1, train_T)
    alphas_cumprod = torch.cos(((steps + s) / (1 + s)) * np.pi / 2) ** 2

    # Choose sampling points equally spaced in log-SNR
    rho = torch.log(alphas_cumprod / (1 - alphas_cumprod))
    rho_target = torch.linspace(rho[-1], rho[0], sampling_T)
    indices = torch.tensor([
      torch.argmin(torch.abs(rho - rt)).item() for rt in rho_target
    ], dtype=torch.long)

  elif schedule == "karras":
    # Log-spaced sigmas between high and low noise
    sigma_min, sigma_max, rho_val = 0.002, 80, 7.0
    i = torch.linspace(0, 1, sampling_T)
    sigmas = (sigma_max ** (1 / rho_val) + i * (sigma_min ** (1 / rho_val) - sigma_max ** (1 / rho_val))) ** rho_val
    sigmas = torch.flip(sigmas, dims=[0])  # high to low noise

    # convert to corresponding alpha_bar values
    alphas_bar_target = 1 / (1 + sigmas**2)

    # map alpha_bar_target to closest training step
    indices = torch.tensor([
      torch.argmin(torch.abs(alphas_bar - ab_t)).item()
      for ab_t in alphas_bar_target
    ], dtype=torch.long)
  else:
    raise ValueError(f"Unknown schedule: {schedule}")

  indices = torch.unique_consecutive(indices)

  print(f"Sampling indices (n={len(indices)}): {indices}")
  return indices

class GaussianDiffusionSampler(nn.Module):
  def __init__(
    self,
    model,
    beta_1,
    beta_T,
    train_T,
    sampling_T=None,
    schedule: str = "linear", # "linear", "cosine", or "karras"
  ):
    super().__init__()

    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T  # default: full schedule

    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    sampling_idx = make_sampling_indices(train_T, sampling_T, schedule, alphas_bar)
    self.sampling_indices = sampling_idx.tolist()

    self.register_buffer("betas", betas[sampling_idx])
    self.register_buffer("alphas", alphas[sampling_idx])
    self.register_buffer("alphas_bar", alphas_bar[sampling_idx])
    self.register_buffer("sqrt_betas", torch.sqrt(betas)[sampling_idx])
    self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar)[sampling_idx])
    self.register_buffer("sqrt_recip_alphas", torch.sqrt(1. / alphas)[sampling_idx])

  def forward(self, x_T, condition):
    x = x_T

    for i, t in enumerate(self.sampling_indices):
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)

      eps = self.model(x, condition, t_vec)

      beta_t = self.betas[i]
      sqrt_beta_t = self.sqrt_betas[i]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]
      sqrt_recip_alpha_t = self.sqrt_recip_alphas[i]

      x = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)

      if t > 0:
        noise = torch.randn_like(x)
        x += sqrt_beta_t * noise

    return x

class DDIMSampler(nn.Module):
  def __init__(
      self,
      model,
      beta_1,
      beta_T,
      train_T,
      sampling_T=None,
      eta=0.0,
      schedule: str = "linear", # "linear", "cosine", or "karras"
    ):
    super().__init__()
    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T  # default: full schedule
    self.eta = eta

    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, 0)

    sampling_idx = make_sampling_indices(train_T, sampling_T, schedule, alphas_bar)
    self.sampling_indices = sampling_idx.tolist()

    alphas_bar = alphas_bar[sampling_idx]
    alphas_bar_prev = torch.cat((alphas_bar[1:], alphas_bar[-1:].clone()))

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
    sigma = torch.zeros_like(alphas_bar)

    if eta > 0:
      frac = (1 - alphas_bar_prev) / (1 - alphas_bar)
      inner = 1 - (alphas_bar / alphas_bar_prev)
      sigma = eta * torch.sqrt(torch.clamp(frac * inner, min=0.0))

    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev - sigma**2, min=0.0))

    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("sigma", sigma)
    self.register_buffer("dir_coeff", dir_coeff)

  def forward(self, x_T, condition):
    x = x_T

    for i, t in enumerate(self.sampling_indices):
      t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
      eps = self.model(x, condition, t_vec)

      sqrt_alpha_bar_t = self.sqrt_alphas_bar[i]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]
      sqrt_alpha_bar_prev = self.sqrt_alphas_bar_prev[i]
      sigma = self.sigma[i]
      dir_coeff = self.dir_coeff[i]

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
      x = sqrt_alpha_bar_prev * x0 + dir_coeff * eps

      if t > 0:
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
    w[j] = num / den

  return w.to(torch.get_default_dtype())  # cast back to default dtype

class DEISSampler(nn.Module):
  def __init__(
    self,
    model,
    beta_1,
    beta_T,
    train_T,
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
    self.method = method
    assert method in ("rhoAB", "tAB"), "method must be 'rhoAB' or 'tAB'"
    self.order = order
    assert self.order >= 0, "order must be >= 0"

    betas = torch.linspace(beta_1, beta_T, self.train_T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    sampling_idx = make_sampling_indices(train_T, sampling_T, schedule, alphas_bar)
    self.sampling_indices = sampling_idx.tolist()

    alphas_bar = alphas_bar[sampling_idx]
    alphas_bar_prev = torch.cat((alphas_bar[1:], alphas_bar[-1:].clone()))

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev, min=0.0))

    self.register_buffer("alphas_bar", alphas_bar)
    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("dir_coeff", dir_coeff)

    if self.method == "rhoAB":
      grid = torch.log(alphas_bar / (1 - alphas_bar))
    else:  # "tAB"
      idxs = torch.tensor(self.sampling_indices, dtype=torch.float32)
      grid = idxs / float(max(1, self.train_T - 1))

    # Precompute Lagrange weights for each step to extrapolate eps from previous points
    # For step i (going from sampling_indices[i] -> sampling_indices[i+1] in time),
    # we'll need weights to evaluate polynomial at time_grid[i+1 - 1] (the next target).
    if use_precomputed and self.order > 0:
      # For each sampling step i (starting from 0), we will compute weights mapping from
      # the available previous times to the *next* target time. The number of available previous
      # points depends on i (warm-up region).
      precomp = []
      for i in range(len(grid)):
        # we are at time t_i = time_grid[i]; we will step to next index j = i+1 (smaller time).
        # But in our loop we iterate from high t to low t, so target time is time_grid[i+1] when available.
        # For the last step there is no next target, but we still precompute an empty list.
        if i + 1 >= len(grid):
          precomp.append(None)
          continue
        # we will use up-to (order+1) previous points: these are at indices [i, i-1, i-2, ...]
        max_m = min(self.order + 1, i + 1)  # can't use more points than available
        # collect previous times in descending order (most recent first)
        prev_indices = [i - j for j in range(max_m)]
        prev_times = grid[prev_indices]  # array of length max_m
        t_target = float(grid[i + 1])   # target is next time (one step further in sampling)
        weights = lagrange_weights(torch.tensor(prev_times), t_target)
        # store weights along with the corresponding prev_indices so we can apply them to eps buffer
        precomp.append((prev_indices, weights))
      # store as attribute (not a buffer since contains lists); convert weights to tensor on-demand
      self._precomputed_weights = precomp
    else:
      self._precomputed_weights = None

  def forward(self, x_T, condition):
    """
    x_T: initial gaussian sample (batch, C, H, W)
    condition: conditioning passed to the model
    returns x_0 estimate
    """
    x = x_T

    # buffer holding past epsilon predictions (eps at times corresponding to sampling_indices)
    eps_buffer = deque(maxlen=self.order + 1)

    # We'll iterate sampling_indices in order (from largest time index to smallest)
    # i loops over positions in the sampling_indices list; at each iteration we are at x = x_{i+1}
    for i, t in enumerate(self.sampling_indices):
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)

      eps = self.model(x, condition, t_vec)
      eps_buffer.append(eps)

      # Decide which epsilon to use for the deterministic DDIM-like update:
      # - if order == 0: use the current eps (standard DDIM deterministic)
      # - if order >=1: extrapolate eps to the target next time using Lagrange weights
      if self.order == 0 or i + 1 >= len(self.sampling_indices):
        # No extrapolation; use current eps and a direct DDIM-like deterministic update
        eps_for_update = eps
      else:
        # extrapolate to next target time (time_grid[i+1]) using up to order+1 previous eps
        if self._precomputed_weights is not None:
          prev_indices, w = self._precomputed_weights[i]
          w = w.to(x.device)
          # prev_indices are indices like [i, i-1, ...] referring to positions in the time_grid
          # but we stored eps_buffer with most recent eps appended each iteration; its length might be less than needed.
          # build a list of eps tensors matching prev_indices ordering:
          # prev_indices are relative to "i" value; map them to positions in eps_buffer:
          # when i=0, eps_buffer contains [eps_at_i], when i=1 contains [eps_at_i, eps_at_i-1], etc.
          # We stored eps in chronological order of iterations; the mapping below selects appropriate entries.
          # create list of tensors matching prev_indices (most recent first)
          eps_list = []
          for rel in prev_indices:
            # rel is i - j where j=0..max_m-1; the element we need is the one appended at iteration rel
            # position in eps_buffer (0=oldest, -1=most recent)
            # mapping: iteration index when that eps was produced = rel. The relative offset from current iteration i:
            offset = i - rel  # e.g., if rel==i then offset=0 -> most recent
            # We want the element at index -(offset+1) from eps_buffer
            idx_in_buffer = - (offset + 1)
            eps_list.append(eps_buffer[idx_in_buffer])
          # stack eps_list as (m, batch, C, H, W) and apply weights w
          stacked = torch.stack(eps_list, dim=0)  # (m, B, C, H, W)
          # reshape w to (m,1,1,1,1) to multiply across tensors
          w_shape = (w.shape[0],) + (1,) * (stacked.ndim - 1)
          w_reshaped = w.view(w_shape)
          eps_for_update = torch.sum(w_reshaped * stacked, dim=0)
        else:
          # no precomputed weights available, fall back to using the most recent eps
          eps_for_update = eps

      sqrt_alpha_bar_t = self.sqrt_alphas_bar[i]
      sqrt_alpha_bar_prev = self.sqrt_alphas_bar_prev[i]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[i]
      dir_coeff = self.dir_coeff[i]

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps_for_update) / sqrt_alpha_bar_t
      x = sqrt_alpha_bar_prev * x0 + dir_coeff * eps_for_update

    return x
