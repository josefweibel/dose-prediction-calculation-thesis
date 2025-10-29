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
  """
  Corrected DEIS (Diffusion Exponential Integrator Sampler) implementation.

  Based on the HuggingFace implementation and the DEIS paper.
  Supports both tAB (time-based Adams-Bashforth) and rhoAB (log-SNR based).
  """

  def __init__(
    self,
    model,
    beta_1: float,
    beta_T: float,
    train_T: int,
    cache_config,
    sampling_T: int = None,
    method: str = "rhoAB",  # "rhoAB" or "tAB"
    order: int = 2,  # 1 for DDIM-like, 2-3 for multistep DEIS
    schedule: str = "linear",
  ):
    super().__init__()
    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T
    self.cache_config = cache_config
    self.method = method
    assert method in ("rhoAB", "tAB"), "method must be 'rhoAB' or 'tAB'"
    self.order = max(1, order)  # Minimum order 1

    # Compute noise schedule
    betas = torch.linspace(beta_1, beta_T, self.train_T).float()
    alphas = 1.0 - betas
    alphas_bar_train = torch.cumprod(alphas, dim=0)

    # Get sampling schedule
    sampling_idx, alphas_bar_t = make_sampling_schedule(
      train_T, self.sampling_T, schedule, alphas_bar_train
    )
    self.sampling_indices = sampling_idx.tolist()

    # Store alpha values
    self.register_buffer("alphas_bar", alphas_bar_t)

    # Compute integration variable based on method
    if self.method == "rhoAB":
      # rhoAB: use log-SNR (rho) as integration variable
      # lambda (log-SNR) = log(alpha_bar / sqrt(1 - alpha_bar))
      lambda_t = torch.log(alphas_bar_t) - 0.5 * torch.log(1.0 - alphas_bar_t)
      self.register_buffer("integration_var", lambda_t)
    else:  # tAB
      # tAB: use normalized time as integration variable
      # Since we're denoising (high t → low t), but lambda increases,
      # we use t/T where t goes from high to low
      time_var = sampling_idx / float(max(1, self.train_T - 1))
      self.register_buffer("integration_var", time_var)

    # Precompute coefficients for each step
    self._precompute_coefficients()

  def _precompute_coefficients(self):
    """
    Precompute the integration coefficients for DEIS multistep updates.
    These are based on integrating the polynomial extrapolation.
    """
    coeffs_list = []

    for i in range(len(self.integration_var)):
      if i + 1 >= len(self.integration_var):
        coeffs_list.append(None)
        continue

      # Current and next integration variable values
      var_s = float(self.integration_var[i])
      var_t = float(self.integration_var[i + 1])
      h = var_t - var_s

      # Get previous values for multistep
      order_used = min(self.order, i + 1)

      if order_used == 1:
        # First-order (DDIM-like): coeff for m_0 only
        coeff = h
        coeffs_list.append(([0], torch.tensor([coeff])))

      elif order_used == 2:
        # Second-order: coeffs for m_0 and m_1
        if i >= 1:
          var_s1 = float(self.integration_var[i - 1])
          h_1 = var_s - var_s1

          # Integration coefficients for 2nd order
          coeff_0 = h + 0.5 * h ** 2 / h_1
          coeff_1 = -0.5 * h ** 2 / h_1

          coeffs_list.append(([0, 1], torch.tensor([coeff_0, coeff_1])))
        else:
          # Fall back to first order
          coeffs_list.append(([0], torch.tensor([h])))

      elif order_used == 3:
        # Third-order: coeffs for m_0, m_1, and m_2
        if i >= 2:
          var_s1 = float(self.integration_var[i - 1])
          var_s2 = float(self.integration_var[i - 2])
          h_1 = var_s - var_s1
          h_2 = var_s - var_s2

          # Integration coefficients for 3rd order
          # These come from integrating the Lagrange polynomial
          coeff_0 = h + 0.5 * h ** 2 / h_1 + (h ** 2) * (2 * h + 3 * h_1) / (6 * h_1 * h_2)
          coeff_1 = -0.5 * h ** 2 / h_1 - (h ** 2) * (h + 2 * h_1) / (2 * h_1 * (h_2 - h_1))
          coeff_2 = (h ** 2) * (h + 2 * h_1) / (6 * h_2 * (h_2 - h_1))

          coeffs_list.append(([0, 1, 2], torch.tensor([coeff_0, coeff_1, coeff_2])))
        elif i >= 1:
          # Fall back to second order
          var_s1 = float(self.integration_var[i - 1])
          h_1 = var_s - var_s1
          coeff_0 = h + 0.5 * h ** 2 / h_1
          coeff_1 = -0.5 * h ** 2 / h_1
          coeffs_list.append(([0, 1], torch.tensor([coeff_0, coeff_1])))
        else:
          # Fall back to first order
          coeffs_list.append(([0], torch.tensor([h])))

    self._coefficients = coeffs_list

  def _convert_to_m(self, eps: torch.Tensor, i: int) -> torch.Tensor:
    """
    Convert epsilon to m (the quantity we integrate in DEIS).
    m = sigma * epsilon = sqrt(1 - alpha_bar) * epsilon
    """
    alpha_bar = self.alphas_bar[i]
    sigma = torch.sqrt(1.0 - alpha_bar)
    return sigma * eps

  def forward(self, x_T, condition):
    x = x_T

    # Buffer to store recent m values (converted epsilon)
    m_buffer = deque(maxlen=self.order)
    cache = DeepCache(self.cache_config)

    for i in range(len(self.sampling_indices)):
      t = self.sampling_indices[i]
      cache.set_current_step(i)

      # Get epsilon prediction
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      eps = self.model(x, condition, t_vec, cache)

      # Convert to m = sigma * eps
      m = self._convert_to_m(eps, i)
      m_buffer.append(m)

      # Check if this is the last step
      if i + 1 >= len(self.sampling_indices):
        break

      # Get precomputed coefficients for this step
      if self._coefficients[i] is None:
        break

      offsets, coeffs = self._coefficients[i]
      coeffs = coeffs.to(x.device)

      # Compute the integral term: sum of coeff_j * m_j
      integral_term = torch.zeros_like(x)
      for offset, coeff in zip(offsets, coeffs):
        # offset 0 = current (m_i), offset 1 = previous (m_{i-1}), etc.
        m_val = m_buffer[-(offset + 1)]  # Access from end of deque
        integral_term += coeff * m_val

      # DEIS update formula in exponential integrator form:
      # x_{t+1} = (alpha_{t+1} / alpha_t) * x_t + alpha_{t+1} * integral_term
      alpha_s = self.alphas_bar[i]
      alpha_t = self.alphas_bar[i + 1]

      x = (alpha_t / alpha_s) * x + alpha_t * integral_term

    return x

