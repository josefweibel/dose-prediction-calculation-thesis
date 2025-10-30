import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import Optional

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

class DEISSampler(nn.Module):
  # uses log-rho space for updates
  # based on diffusers
  def __init__(
    self,
    model: nn.Module,
    beta_1: float,
    beta_T: float,
    train_T: int,
    cache_config,
    sampling_T: Optional[int] = None,
    order: int = 2,
    schedule: str = "linear",
  ):
    super().__init__()

    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T if sampling_T is not None else train_T
    self.order = order
    self.schedule = schedule
    self.cache_config = cache_config

    betas = torch.linspace(beta_1, beta_T, train_T, dtype=torch.float32)

    # Compute training alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Store training schedule as buffers
    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alphas_cumprod', alphas_cumprod)

    if schedule != 'linear':
      raise NotImplementedError("Only 'linear' schedule is implemented for DEISSampler.")

    # Precompute sampling schedule
    sampling_indices, alphas_bar_sampling = make_sampling_schedule(
      train_T, self.sampling_T, schedule, alphas_cumprod
    )

    # Store sampling timesteps as buffer
    self.register_buffer('timesteps', sampling_indices)

    sigmas = ((1 - alphas_bar_sampling) / alphas_bar_sampling) ** 0.5
    sigmas = torch.cat([sigmas, torch.zeros(1)]).float()
    self.register_buffer('sigmas', sigmas)

  def __sigma_to_alpha_sigma_t(self, sigma):
    alpha_t = 1 / ((sigma**2 + 1) ** 0.5)
    sigma_t = sigma * alpha_t

    return alpha_t, sigma_t

  def __deis_first_order_update(
    self,
    x: torch.Tensor,
    step_idx: int,
    model_outputs: list
  ) -> torch.Tensor:
    sigma_t, sigma_s = self.sigmas[step_idx + 1], self.sigmas[step_idx]
    alpha_t, sigma_t = self.__sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = self.__sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)
    h = lambda_t - lambda_s

    m0 = model_outputs[-1]
    return (alpha_t / alpha_s) * x - (sigma_t * (torch.exp(h) - 1.0)) * m0

  def __deis_second_order_update(
    self,
    x: torch.Tensor,
    step_idx: int,
    model_outputs: list
  ) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1 = (
      self.sigmas[step_idx + 1],
      self.sigmas[step_idx],
      self.sigmas[step_idx - 1],
    )

    alpha_t, sigma_t = self.__sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self.__sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self.__sigma_to_alpha_sigma_t(sigma_s1)

    m0, m1 = model_outputs[-1], model_outputs[-2]
    rho_t, rho_s0, rho_s1 = sigma_t / alpha_t, sigma_s0 / alpha_s0, sigma_s1 / alpha_s1

    def ind_fn(t, b, c):
      # Integrate[(log(t) - log(c)) / (log(b) - log(c)), {t}]
      return t * (-torch.log(c) + torch.log(t) - 1) / (torch.log(b) - torch.log(c))

    coef1 = ind_fn(rho_t, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s0, rho_s1)
    coef2 = ind_fn(rho_t, rho_s1, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s0)

    return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1)

  def __deis_third_order_update(
    self,
    x: torch.Tensor,
    step_idx: int,
    model_outputs: list
  ) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1, sigma_s2 = (
      self.sigmas[step_idx + 1],
      self.sigmas[step_idx],
      self.sigmas[step_idx - 1],
      self.sigmas[step_idx - 2],
    )

    alpha_t, sigma_t = self.__sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = self.__sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = self.__sigma_to_alpha_sigma_t(sigma_s1)
    alpha_s2, sigma_s2 = self.__sigma_to_alpha_sigma_t(sigma_s2)

    m0, m1, m2 = model_outputs[-1], model_outputs[-2], model_outputs[-3]

    rho_t, rho_s0, rho_s1, rho_s2 = (
      sigma_t / alpha_t,
      sigma_s0 / alpha_s0,
      sigma_s1 / alpha_s1,
      sigma_s2 / alpha_s2,
    )

    def ind_fn(t, b, c, d):
      # Integrate[(log(t) - log(c))(log(t) - log(d)) / (log(b) - log(c))(log(b) - log(d)), {t}]
      numerator = t * (
        torch.log(c) * (torch.log(d) - torch.log(t) + 1)
        - torch.log(d) * torch.log(t)
        + torch.log(d)
        + torch.log(t) ** 2
        - 2 * torch.log(t)
        + 2
      )
      denominator = (torch.log(b) - torch.log(c)) * (torch.log(b) - torch.log(d))
      return numerator / denominator

    coef1 = ind_fn(rho_t, rho_s0, rho_s1, rho_s2) - ind_fn(rho_s0, rho_s0, rho_s1, rho_s2)
    coef2 = ind_fn(rho_t, rho_s1, rho_s2, rho_s0) - ind_fn(rho_s0, rho_s1, rho_s2, rho_s0)
    coef3 = ind_fn(rho_t, rho_s2, rho_s0, rho_s1) - ind_fn(rho_s0, rho_s2, rho_s0, rho_s1)

    return alpha_t * (x / alpha_s0 + coef1 * m0 + coef2 * m1 + coef3 * m2)

  def __step(
    self,
    eps: torch.Tensor,
    step_idx: int,
    x: torch.Tensor,
    model_outputs: list,
  ) -> torch.Tensor:
    for i in range(self.order - 1):
      model_outputs[i] = model_outputs[i + 1]
    model_outputs[-1] = eps

    lower_order_final = (step_idx == len(self.timesteps) - 1) and len(self.timesteps) < 15
    lower_order_second = (step_idx == len(self.timesteps) - 2) and len(self.timesteps) < 15

    n_cached = sum([1 for m in model_outputs if m is not None])
    if self.order == 1 or n_cached == 1 or lower_order_final:
      # First-order update (DDIM-like with exponential integrator)
      return self.__deis_first_order_update(x, step_idx, model_outputs)
    elif self.order == 2 or n_cached == 2 or lower_order_second:
      return self.__deis_second_order_update(x, step_idx, model_outputs)
    else:
      return self.__deis_third_order_update(x, step_idx, model_outputs)

  @torch.no_grad()
  def forward(self, x_T: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    """
    Full sampling process from noise to clean sample.

    Args:
      x_T: Initial noise tensor [B, C, H, W]
      condition: Conditioning information (required) [B, D]

    Returns:
      Denoised sample x_0 [B, C, H, W]
    """
    x = x_T
    cache = DeepCache(self.cache_config)
    model_outputs = [None] * self.order

    for step_idx, t in enumerate(self.timesteps):
      cache.set_current_step(step_idx)
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      model_output = self.model(x, condition, t_vec, cache)
      x = self.__step(model_output, step_idx, x, model_outputs)

    return x

