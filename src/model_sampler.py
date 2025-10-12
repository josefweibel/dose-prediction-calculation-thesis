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
      order=sampling_config['order'],
      parameterization='epsilon',
      schedule=sampling_config['schedule'],
      use_precomputed=True,
    )
  else:
    raise ValueError(f"Unknown sampler: {sampling_config['sampler']}")


def make_sampling_indices(train_T, sampling_T, schedule, beta_1, beta_T):
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
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]  # normalize

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
    sigmas = torch.flip(sigmas, dims=[0])

    # Match sigma levels to closest training timestep
    betas = torch.linspace(beta_1, beta_T, train_T)
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    noise_levels = torch.sqrt((1 - alphas_bar) / alphas_bar)

    indices = torch.tensor([
      torch.argmin(torch.abs(noise_levels - s)).item() for s in sigmas
    ], dtype=torch.long)

  else:
    raise ValueError(f"Unknown schedule: {schedule}")

  return indices.tolist()

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

    self.sampling_indices = make_sampling_indices(train_T, sampling_T, schedule, beta_1, beta_T)

    self.register_buffer("betas", betas)
    self.register_buffer("alphas", alphas)
    self.register_buffer("alphas_bar", alphas_bar)
    self.register_buffer("sqrt_betas", torch.sqrt(betas))
    self.register_buffer("sqrt_one_minus_alphas_bar", torch.sqrt(1. - alphas_bar))
    self.register_buffer("sqrt_recip_alphas", torch.sqrt(1. / alphas))

  def forward(self, x_T, condition):
    x = x_T

    for t in self.sampling_indices:
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)

      eps = self.model(x, condition, t_vec)

      beta_t = self.betas[t]
      sqrt_beta_t = self.sqrt_betas[t]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
      sqrt_recip_alpha_t = self.sqrt_recip_alphas[t]

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
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:train_T]

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
    sigma = torch.zeros_like(alphas_bar)

    if eta > 0:
      frac = (1 - alphas_bar_prev) / (1 - alphas_bar)
      inner = 1 - (alphas_bar / alphas_bar_prev)
      sigma = eta * torch.sqrt(torch.clamp(frac * inner, min=0.0))

    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev - sigma**2, min=0.0))

    self.sampling_indices = make_sampling_indices(train_T, sampling_T, schedule, beta_1, beta_T)

    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("sigma", sigma)
    self.register_buffer("dir_coeff", dir_coeff)

  def forward(self, x_T, condition):
    x = x_T

    for t in self.sampling_indices:
      t_vec = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
      eps = self.model(x, condition, t_vec)

      sqrt_alpha_bar_t = self.sqrt_alphas_bar[t]
      sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[t]
      sqrt_a_bar_prev = self.sqrt_alphas_bar_prev[t]
      sigma = self.sigma[t]
      dir_coeff = self.dir_coeff[t]

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t
      x = sqrt_a_bar_prev * x0 + dir_coeff * eps

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
    order: int = 0, # 0 == DDIM-like; 1..k for multistep extrapolation
    parameterization: str = "epsilon", # "epsilon" (default), "v" or "x0"
    schedule: str = "linear", # "linear", "cosine", or "karras"
    use_precomputed: bool = True,
  ):
    super().__init__()
    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T
    self.order = order
    assert self.order >= 0, "order must be >= 0"
    self.parameterization = parameterization

    betas = torch.linspace(beta_1, beta_T, self.train_T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:train_T]

    sqrt_alphas_bar = torch.sqrt(alphas_bar)
    sqrt_one_minus_alphas_bar = torch.sqrt(1 - alphas_bar)
    sqrt_alphas_bar_prev = torch.sqrt(alphas_bar_prev)
    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev, min=0.0))

    self.sampling_indices = make_sampling_indices(train_T, sampling_T, schedule, beta_1, beta_T)

    self.register_buffer("alphas_bar", alphas_bar)
    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("dir_coeff", dir_coeff)

    # Precompute normalized time grid in [0,1] for the selected indices.
    # We use a simple mapping: t = idx / (train_T - 1).
    idxs = torch.tensor(self.sampling_indices, dtype=torch.float32)
    time_grid = idxs / float(max(1, self.train_T - 1))

    # Precompute Lagrange weights for each step to extrapolate eps from previous points
    # For step i (going from sampling_indices[i] -> sampling_indices[i+1] in time),
    # we'll need weights to evaluate polynomial at time_grid[i+1 - 1] (the next target).
    if use_precomputed and self.order > 0:
      # For each sampling step i (starting from 0), we will compute weights mapping from
      # the available previous times to the *next* target time. The number of available previous
      # points depends on i (warm-up region).
      precomp = []
      times = time_grid.numpy()
      for i in range(len(time_grid)):
        # we are at time t_i = time_grid[i]; we will step to next index j = i+1 (smaller time).
        # But in our loop we iterate from high t to low t, so target time is time_grid[i+1] when available.
        # For the last step there is no next target, but we still precompute an empty list.
        if i + 1 >= len(time_grid):
          precomp.append(None)
          continue
        # we will use up-to (order+1) previous points: these are at indices [i, i-1, i-2, ...]
        max_m = min(self.order + 1, i + 1)  # can't use more points than available
        # collect previous times in descending order (most recent first)
        prev_indices = [i - j for j in range(max_m)]
        prev_times = times[prev_indices]  # array of length max_m
        t_target = float(times[i + 1])   # target is next time (one step further in sampling)
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
    for i, idx in enumerate(self.sampling_indices):
      t_vec = torch.full((x.shape[0],), idx, dtype=torch.long, device=x.device)

      # model output
      model_out = self.model(x, condition, t_vec)
      # convert model_out to epsilon if necessary
      if self.parameterization == "epsilon":
        eps = model_out
      elif self.parameterization == "x0":
        # model returns x0 -> convert to eps consistent with notation:
        # eps = (x - sqrt(alpha_bar_t)*x0) / sqrt(1 - alpha_bar_t)
        alpha_bar_t = self.alphas_bar[idx]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_bar[idx]
        eps = (x - math.sqrt(alpha_bar_t) * model_out) / max(1e-12, sqrt_one_minus_alpha_bar_t)
      elif self.parameterization == "v":
        # v = alpha_t * eps - sigma_t * x0  (paper convention); we invert to eps
        # Here we assume v = model_out; use alphas_bar to build alpha_t and sigma_t approx
        # For simplicity use sqrt(alpha_bar) as alpha_t (approx) and sigma = sqrt(1-alpha_bar)
        alpha_bar_t = self.alphas_bar[idx]
        alpha_t = math.sqrt(alpha_bar_t)
        sigma_t = math.sqrt(max(1e-12, 1.0 - alpha_bar_t))
        v = model_out
        # from v = alpha_t * eps - sigma_t * x0  and x = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps
        # solve linear system for eps: (this uses algebra; rearranged for eps)
        # express x0 from v and eps: x0 = (alpha_t*eps - v)/sigma_t
        # plug into x equation: x = sqrt(alpha_bar)*(alpha_t*eps - v)/sigma_t + sqrt(1-alpha_bar)*eps
        # solve for eps: eps * (sqrt(alpha_bar)*alpha_t/sigma_t + sqrt(1-alpha_bar)) = x + sqrt(alpha_bar)*v/sigma_t
        lhs = (math.sqrt(alpha_bar_t) * alpha_t / (sigma_t + 1e-12) + math.sqrt(max(1e-12, 1.0 - alpha_bar_t)))
        rhs = x + (math.sqrt(alpha_bar_t) * v) / (sigma_t + 1e-12)
        eps = rhs / (lhs + 1e-12)
      else:
        raise ValueError("Unknown parameterization")

      # push the most recent eps onto buffer (most recent at right)
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
