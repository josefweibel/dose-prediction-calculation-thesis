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

class DEISSampler(nn.Module):
  def __init__(
    self,
    model: nn.Module,
    beta_1: float,
    beta_T: float,
    train_T: int,
    cache_config,
    sampling_T: Optional[int] = None,
    method: str = "rhoAB",
    order: int = 2,
    schedule: str = "linear",
  ):
    super().__init__()

    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T if sampling_T is not None else train_T
    self.method = method.lower()
    self.order = order
    self.schedule = schedule
    self.cache_config = cache_config

    # Note: Following HuggingFace Diffusers, only 'rhoab' (log-rho) is properly supported
    # The 'tab' method is kept for API compatibility but uses the same implementation
    if self.method not in ['tab', 'rhoab']:
      raise ValueError(f"method must be 'tAB' or 'rhoAB', got {method}")

    # Create beta schedule - ALWAYS use the training schedule (typically linear)
    # The 'schedule' parameter only affects how we select timesteps during sampling,
    # NOT the underlying beta schedule the model was trained on
    betas = torch.linspace(beta_1, beta_T, train_T, dtype=torch.float32)

    # Compute training alphas
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    # Store training schedule as buffers
    self.register_buffer('betas', betas)
    self.register_buffer('alphas', alphas)
    self.register_buffer('alphas_cumprod', alphas_cumprod)

    # Precompute sampling schedule
    sampling_indices, sampling_alphas_bar = make_sampling_schedule(
      train_T, self.sampling_T, schedule, alphas_cumprod
    )

    # Store sampling timesteps as buffer
    self.register_buffer('timesteps', sampling_indices)
    self.register_buffer('sampling_alphas_bar', sampling_alphas_bar)

    # Precompute derived quantities for sampling
    # Store actual alpha and sigma (not square roots) as DEIS requires
    alpha_t = torch.sqrt(sampling_alphas_bar)  # α_t = sqrt(ᾱ_t)
    sigma_t = torch.sqrt(1 - sampling_alphas_bar)  # σ_t = sqrt(1 - ᾱ_t)

    # Log-SNR: λ = log(α/σ)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)

    # Also compute rho = σ/α for coefficient calculation
    rho_t = sigma_t / alpha_t

    self.register_buffer('alpha_t', alpha_t)
    self.register_buffer('sigma_t', sigma_t)
    self.register_buffer('lambda_t', lambda_t)
    self.register_buffer('rho_t', rho_t)

    # Precompute coefficients for all pairs of timesteps
    self._precompute_coefficients()

    # Cache for model outputs (for multistep methods)
    self.model_outputs = []

  def _compute_ind_fn_2(self, rho_t, rho_b, rho_c):
    """
    Compute the integral of Lagrange basis polynomial for second-order DEIS.

    Integrates: ∫ (log(ρ) - log(ρ_c)) / (log(ρ_b) - log(ρ_c)) dρ

    Result: ρ * (-log(ρ_c) + log(ρ) - 1) / (log(ρ_b) - log(ρ_c))
    """
    numerator = rho_t * (-torch.log(torch.tensor(rho_c)) + torch.log(torch.tensor(rho_t)) - 1)
    denominator = torch.log(torch.tensor(rho_b)) - torch.log(torch.tensor(rho_c))
    return (numerator / denominator).item()

  def _compute_ind_fn_3(self, rho_t, rho_b, rho_c, rho_d):
    """
    Compute the integral of Lagrange basis polynomial for third-order DEIS.

    Integrates the quadratic Lagrange basis over log-rho space.
    """
    log_b = torch.log(torch.tensor(rho_b))
    log_c = torch.log(torch.tensor(rho_c))
    log_d = torch.log(torch.tensor(rho_d))
    log_t = torch.log(torch.tensor(rho_t))

    numerator = rho_t * (
      log_c * (log_d - log_t + 1)
      - log_d * log_t
      + log_d
      + log_t ** 2
      - 2 * log_t
      + 2
    )
    denominator = (log_b - log_c) * (log_b - log_d)
    return (numerator / denominator).item()

  def _precompute_coefficients(self):
    """
    Precompute multistep coefficients for all timestep pairs.

    Coefficients are computed by integrating Lagrange polynomial basis functions
    in log-rho space, following the HuggingFace Diffusers DEIS implementation.

    Key insight: timesteps decrease [999, ..., 0] but rho = σ/α INCREASES during
    denoising (from low to high). The Lagrange integrals compute ∫_{rho_s}^{rho_t}
    where rho_s < rho_t, stepping from higher noise to lower noise.
    """
    T = len(self.timesteps)

    # Storage for precomputed coefficients
    # For step i: coef_matrix[i, j] = coefficient for j-th most recent output
    # j=0 is current, j=1 is previous, j=2 is two steps ago
    max_order = min(self.order, 3)  # Support up to order 3
    self.coef_matrix = torch.zeros(T, max_order, dtype=torch.float32)

    for i in range(T - 1):
      # Get rho values for current and next timesteps
      # timesteps: [t_0=999, t_1=888, ..., t_N=0]  (DECREASING)
      # rho:       [rho_0,   rho_1,   ..., rho_N]  (INCREASING, since rho = σ/α)
      # At step i: we're at timestep[i] and moving to timestep[i+1]
      rho_s = self.rho_t[i].item()      # Current timestep (where we are now)
      rho_t = self.rho_t[i + 1].item()  # Next timestep (where we're stepping to)

      # Order 1: First-order (DDIM-like) - uses only current output
      # Coefficient is implicitly 1, handled separately in step()

      # Order 2: Second-order multistep (needs one previous output)
      if i >= 1 and self.order >= 2:
        rho_s_prev = self.rho_t[i - 1].item()  # Previous timestep

        # Compute coefficients using integrated Lagrange polynomials
        # Integration is from rho_s to rho_t (current to next)
        coef_curr = self._compute_ind_fn_2(rho_t, rho_s, rho_s_prev) - \
                    self._compute_ind_fn_2(rho_s, rho_s, rho_s_prev)
        coef_prev = self._compute_ind_fn_2(rho_t, rho_s_prev, rho_s) - \
                    self._compute_ind_fn_2(rho_s, rho_s_prev, rho_s)

        self.coef_matrix[i, 0] = coef_curr
        self.coef_matrix[i, 1] = coef_prev

      # Order 3: Third-order multistep (needs two previous outputs)
      if i >= 2 and self.order >= 3:
        rho_s_prev = self.rho_t[i - 1].item()
        rho_s_prev_prev = self.rho_t[i - 2].item()

        # Compute coefficients using integrated Lagrange polynomials
        coef_curr = self._compute_ind_fn_3(rho_t, rho_s, rho_s_prev, rho_s_prev_prev) - \
                    self._compute_ind_fn_3(rho_s, rho_s, rho_s_prev, rho_s_prev_prev)
        coef_prev = self._compute_ind_fn_3(rho_t, rho_s_prev, rho_s, rho_s_prev_prev) - \
                    self._compute_ind_fn_3(rho_s, rho_s_prev, rho_s, rho_s_prev_prev)
        coef_prev_prev = self._compute_ind_fn_3(rho_t, rho_s_prev_prev, rho_s, rho_s_prev) - \
                         self._compute_ind_fn_3(rho_s, rho_s_prev_prev, rho_s, rho_s_prev)

        self.coef_matrix[i, 0] = coef_curr
        self.coef_matrix[i, 1] = coef_prev
        self.coef_matrix[i, 2] = coef_prev_prev

    self.register_buffer('coef_matrix_buf', self.coef_matrix)

    # DEBUG: Print first few rows of coefficient matrix
    print(f"\n[DEIS DEBUG] Coefficient matrix (first 5 steps, order {self.order}):")
    for i in range(min(5, T-1)):
      print(f"  Step {i}: coefs = {self.coef_matrix[i, :min(self.order, 3)].tolist()}")
    print(f"  rho values (first 5): {self.rho_t[:5].tolist()}")

  def step(
    self,
    model_output: torch.Tensor,
    step_idx: int,
    sample: torch.Tensor
  ) -> torch.Tensor:
    """
    Predict the sample at the next timestep using DEIS multistep method.

    Args:
      model_output: Direct output from learned diffusion model (epsilon prediction)
      step_idx: Index in the sampling schedule (0 to sampling_T-1)
      sample: Current instance of sample being created

    Returns:
      Predicted sample at next timestep
    """
    # Get current timestep parameters
    alpha_curr = self.alpha_t[step_idx]
    sigma_curr = self.sigma_t[step_idx]

    # Convert epsilon prediction to DEIS representation
    # First: epsilon -> x0 prediction
    x0_pred = (sample - sigma_curr * model_output) / alpha_curr

    # Second: x0 -> DEIS model output (noise component)
    # model_output_deis = (x - α*x0) / σ
    deis_output = (sample - alpha_curr * x0_pred) / sigma_curr

    # DEBUG: Check if deis_output equals model_output (epsilon)
    if step_idx < 3:
      diff = torch.abs(deis_output - model_output).max().item()
      print(f"[DEIS DEBUG] Step {step_idx}: deis_output vs epsilon diff = {diff:.6e}")

    # Determine effective order based on available history
    num_prev = len(self.model_outputs)
    eff_order = min(self.order, num_prev + 1, 3)

    # Get next timestep parameters
    if step_idx < len(self.timesteps) - 1:
      alpha_next = self.alpha_t[step_idx + 1]
      sigma_next = self.sigma_t[step_idx + 1]
    else:
      # Final step: go to t=0 (clean image)
      alpha_next = torch.tensor(1.0, device=sample.device, dtype=sample.dtype)
      sigma_next = torch.tensor(0.0, device=sample.device, dtype=sample.dtype)

    # Apply DEIS update based on order
    if eff_order == 1 or num_prev == 0:
      # First-order update (DDIM-like with exponential integrator)
      lambda_curr = self.lambda_t[step_idx]
      if step_idx < len(self.timesteps) - 1:
        lambda_next = self.lambda_t[step_idx + 1]
        h = lambda_next - lambda_curr
        # x_next = (α_next/α_curr) * x - σ_next * (exp(h) - 1) * model_output
        next_sample = (alpha_next / alpha_curr) * sample - \
                      (sigma_next * (torch.exp(h) - 1.0)) * deis_output
      else:
        # Final step
        next_sample = alpha_next * x0_pred

    else:
      # Multistep update (order 2 or 3)
      # Combine current and previous outputs using precomputed coefficients
      combined_coef = self.coef_matrix_buf[step_idx, 0] * deis_output

      # DEBUG: Print coefficients and history usage
      if step_idx < 5:
        print(f"[DEIS DEBUG] Step {step_idx}, Order {eff_order}:")
        print(f"  coef[{step_idx},0] = {self.coef_matrix_buf[step_idx, 0]:.6f}")
        print(f"  deis_output mean = {deis_output.mean().item():.6e}")
        print(f"  num_prev = {num_prev}, history size = {len(self.model_outputs)}")

      for i in range(min(num_prev, eff_order - 1)):
        # Access from end of list (most recent first)
        hist_output = self.model_outputs[-(i+1)]
        coef = self.coef_matrix_buf[step_idx, i + 1]
        combined_coef += coef * hist_output

        # DEBUG
        if step_idx < 5:
          print(f"  coef[{step_idx},{i+1}] = {coef:.6f}, hist[{-(i+1)}] mean = {hist_output.mean().item():.6e}")

      # DEIS multistep formula: x_next = α_next * (x/α_curr + combined_coef)
      next_sample = alpha_next * (sample / alpha_curr + combined_coef)

      # DEBUG
      if step_idx < 5:
        print(f"  combined_coef mean = {combined_coef.mean().item():.6e}")

    # Store model output for future steps (FIFO: append to end)
    self.model_outputs.append(deis_output)
    if len(self.model_outputs) > self.order:
      self.model_outputs.pop(0)  # Remove oldest

    return next_sample

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
    self.model_outputs = []
    x = x_T
    cache = DeepCache(self.cache_config)

    for step_idx, t in enumerate(self.timesteps):
      cache.set_current_step(step_idx)
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x.device)
      model_output = self.model(x, condition, t_vec, cache)
      x = self.step(model_output, step_idx, x)

    return x
