"""
Diffusion utilities for parameterization conversions and loss weighting.

Supports both epsilon (noise prediction) and v-prediction parameterizations
as described in "Progressive Distillation for Fast Sampling of Diffusion Models"
(Salimans & Ho, ICLR 2022).

Notation:
  - alpha_t = sqrt(alpha_bar_t), the signal coefficient
  - sigma_t = sqrt(1 - alpha_bar_t), the noise coefficient
  - x_0: clean data
  - x_t: noisy data at timestep t
  - epsilon: noise added to create x_t
  - v: velocity, defined as v = alpha_t * epsilon - sigma_t * x_0
"""

import torch


def compute_noise_schedule(beta_1: float, beta_T: float, T: int, device=None):
  """
  Compute the noise schedule parameters.

  Args:
    beta_1: Starting noise level
    beta_T: Ending noise level
    T: Number of timesteps

  Returns:
    Dictionary containing:
    - betas: [T] noise schedule
    - alphas: [T] = 1 - betas
    - alphas_bar: [T] cumulative product of alphas
    - sqrt_alphas_bar: [T] signal coefficient (alpha_t)
    - sqrt_one_minus_alphas_bar: [T] noise coefficient (sigma_t)
  """
  betas = torch.linspace(beta_1, beta_T, T, device=device).float()
  alphas = 1.0 - betas
  alphas_bar = torch.cumprod(alphas, dim=0)

  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_bar': alphas_bar,
    'sqrt_alphas_bar': torch.sqrt(alphas_bar),
    'sqrt_one_minus_alphas_bar': torch.sqrt(1.0 - alphas_bar),
  }


def epsilon_to_x0(epsilon, x_t, alpha_t, sigma_t):
  """
  Convert noise prediction to x0 prediction.

  x_t = alpha_t * x_0 + sigma_t * epsilon
  => x_0 = (x_t - sigma_t * epsilon) / alpha_t

  Args:
    epsilon: Predicted noise [B, ...]
    x_t: Noisy sample [B, ...]
    alpha_t: Signal coefficient (sqrt_alpha_bar_t) [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient (sqrt_one_minus_alpha_bar_t) [B, 1, 1, ...] or scalar

  Returns:
    x0: Predicted clean sample [B, ...]
  """
  return (x_t - sigma_t * epsilon) / alpha_t


def v_to_x0(v, x_t, alpha_t, sigma_t):
  """
  Convert v-prediction to x0 prediction.

  v = alpha_t * epsilon - sigma_t * x_0
  x_t = alpha_t * x_0 + sigma_t * epsilon

  From these two equations:
  x_0 = alpha_t * x_t - sigma_t * v

  Args:
    v: Predicted velocity [B, ...]
    x_t: Noisy sample [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar

  Returns:
    x0: Predicted clean sample [B, ...]
  """
  return alpha_t * x_t - sigma_t * v


def v_to_epsilon(v, x_t, alpha_t, sigma_t):
  """
  Convert v-prediction to epsilon prediction.

  v = alpha_t * epsilon - sigma_t * x_0
  x_t = alpha_t * x_0 + sigma_t * epsilon

  From these: epsilon = alpha_t * v + sigma_t * (x_t / (alpha_t^2 + sigma_t^2))
  Since alpha_t^2 + sigma_t^2 = 1: epsilon = alpha_t * v + sigma_t * x_t

  Args:
    v: Predicted velocity [B, ...]
    x_t: Noisy sample [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar

  Returns:
    epsilon: Predicted noise [B, ...]
  """
  return alpha_t * v + sigma_t * x_t


def x0_to_epsilon(x0, x_t, alpha_t, sigma_t):
  """
  Convert x0 prediction to epsilon prediction.

  x_t = alpha_t * x_0 + sigma_t * epsilon
  => epsilon = (x_t - alpha_t * x_0) / sigma_t

  Args:
    x0: Predicted clean sample [B, ...]
    x_t: Noisy sample [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar

  Returns:
    epsilon: Predicted noise [B, ...]
  """
  return (x_t - alpha_t * x0) / sigma_t


def x0_to_v(x0, epsilon, alpha_t, sigma_t):
  """
  Convert x0 and epsilon to v-prediction.

  v = alpha_t * epsilon - sigma_t * x_0

  Args:
    x0: Clean sample [B, ...]
    epsilon: Noise [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar

  Returns:
    v: Velocity [B, ...]
  """
  return alpha_t * epsilon - sigma_t * x0


def compute_snr_weights(alpha_t, sigma_t, weighting='snr_plus_one'):
  """
  Compute loss weights based on signal-to-noise ratio.

  SNR(t) = alpha_t^2 / sigma_t^2

  Args:
    alpha_t: Signal coefficient [T] or [B, 1, 1, ...]
    sigma_t: Noise coefficient [T] or [B, 1, 1, ...]
    weighting: Type of weighting
      - 'uniform': w(t) = 1
      - 'snr': w(t) = SNR(t)
      - 'snr_plus_one': w(t) = SNR(t) + 1 (recommended for v-prediction)
      - 'truncated_snr': w(t) = max(SNR(t), 1)

  Returns:
    weights: Loss weights with same shape as alpha_t
  """
  snr = (alpha_t ** 2) / (sigma_t ** 2 + 1e-8)

  if weighting == 'uniform':
    return torch.ones_like(snr)
  elif weighting == 'snr':
    return snr
  elif weighting == 'snr_plus_one':
    return snr + 1.0
  elif weighting == 'truncated_snr':
    return torch.clamp(snr, min=1.0)
  else:
    raise ValueError(f"Unknown weighting scheme: {weighting}")


def predict_x0(model_output, x_t, alpha_t, sigma_t, parameterization='epsilon'):
  """
  Unified x0 prediction from model output.

  Args:
    model_output: Model prediction [B, ...]
    x_t: Noisy sample [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar
    parameterization: 'epsilon' or 'v'

  Returns:
    x0: Predicted clean sample [B, ...]
  """
  if parameterization == 'epsilon':
    return epsilon_to_x0(model_output, x_t, alpha_t, sigma_t)
  elif parameterization == 'v':
    return v_to_x0(model_output, x_t, alpha_t, sigma_t)
  else:
    raise ValueError(f"Unknown parameterization: {parameterization}")


def compute_training_target(x0, epsilon, alpha_t, sigma_t, parameterization='epsilon'):
  """
  Compute the training target for the given parameterization.

  Args:
    x0: Clean sample [B, ...]
    epsilon: Noise [B, ...]
    alpha_t: Signal coefficient [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient [B, 1, 1, ...] or scalar
    parameterization: 'epsilon' or 'v'

  Returns:
    target: Training target [B, ...]
  """
  if parameterization == 'epsilon':
    return epsilon
  elif parameterization == 'v':
    return x0_to_v(x0, epsilon, alpha_t, sigma_t)
  else:
    raise ValueError(f"Unknown parameterization: {parameterization}")


def ddim_step(x_t, x0_pred, alpha_t, sigma_t, alpha_s, sigma_s, eta=0.0):
  """
  Perform a single DDIM step from timestep t to timestep s (t > s).

  DDIM update (Eq. 7-8 from progressive distillation paper):
  z_s = alpha_s * x0_pred + sqrt(1 - alpha_s^2 - sigma_ddim^2) * eps_pred + sigma_ddim * noise

  For deterministic sampling (eta=0):
  z_s = alpha_s * x0_pred + sigma_s * (x_t - alpha_t * x0_pred) / sigma_t

  Args:
    x_t: Current noisy sample [B, ...]
    x0_pred: Predicted clean sample [B, ...]
    alpha_t: Signal coefficient at t [B, 1, 1, ...] or scalar
    sigma_t: Noise coefficient at t [B, 1, 1, ...] or scalar
    alpha_s: Signal coefficient at s [B, 1, 1, ...] or scalar
    sigma_s: Noise coefficient at s [B, 1, 1, ...] or scalar
    eta: Stochasticity parameter (0 = deterministic DDIM)

  Returns:
    x_s: Sample at timestep s [B, ...]
  """
  # Compute epsilon from x0_pred
  eps_pred = (x_t - alpha_t * x0_pred) / sigma_t

  if eta > 0:
    # Stochastic DDIM
    sigma_ddim = eta * torch.sqrt((sigma_s ** 2 / sigma_t ** 2) * (1 - alpha_t ** 2 / alpha_s ** 2))
    sigma_ddim = torch.clamp(sigma_ddim, min=0.0)
    dir_coeff = torch.sqrt(torch.clamp(sigma_s ** 2 - sigma_ddim ** 2, min=0.0))
    noise = torch.randn_like(x_t)
    x_s = alpha_s * x0_pred + dir_coeff * eps_pred + sigma_ddim * noise
  else:
    # Deterministic DDIM
    x_s = alpha_s * x0_pred + sigma_s * eps_pred

  return x_s


def compute_distillation_target(z_t, z_t_double_prime, alpha_t, sigma_t,
                                alpha_t_double_prime, sigma_t_double_prime):
  """
  Compute the distillation target x_tilde from the teacher's two-step prediction.

  From Equation 43 in the progressive distillation paper:
  The student needs to predict x_tilde such that DDIM from t to t'' with x_tilde
  gives z_t''. We invert this to find x_tilde:

  z_t'' = alpha_t'' * x_tilde + sigma_t'' * (z_t - alpha_t * x_tilde) / sigma_t
  => x_tilde = (z_t'' - (sigma_t'' / sigma_t) * z_t) / (alpha_t'' - (sigma_t'' / sigma_t) * alpha_t)

  Args:
    z_t: Noisy sample at timestep t [B, ...]
    z_t_double_prime: Teacher's prediction at t'' (after 2 DDIM steps) [B, ...]
    alpha_t: Signal coefficient at t
    sigma_t: Noise coefficient at t
    alpha_t_double_prime: Signal coefficient at t''
    sigma_t_double_prime: Noise coefficient at t''

  Returns:
    x_tilde: Distillation target [B, ...]
  """
  sigma_ratio = sigma_t_double_prime / (sigma_t + 1e-8)
  numerator = z_t_double_prime - sigma_ratio * z_t
  denominator = alpha_t_double_prime - sigma_ratio * alpha_t

  return numerator / (denominator + 1e-8)
