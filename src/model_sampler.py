import torch
from torch import nn
from torch.nn import functional as F


def build_sampler(model_config, sampling_config, model):
  if sampling_config['sampler'] == 'ddpm':
    return GaussianDiffusionSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      sampling_config['T'],
    )
  elif sampling_config['sampler'] == 'ddim':
    return DDIMSampler(
      model,
      model_config['beta_1'],
      model_config['beta_T'],
      model_config['T'],
      sampling_config['T'],
      eta=sampling_config['eta']
    )
  else:
    raise ValueError(f"Unknown sampler: {sampling_config['sampler']}")


def extract(v, t, x_shape):
  """
  Extract some coefficients at specified timesteps, then reshape to
  [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
  """
  out = v.to(t.device).gather(0, t).float()
  return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionSampler(nn.Module):
  def __init__(self, model, beta_1, beta_T, train_T, sampling_T=None):
    super().__init__()

    self.model = model
    self.train_T = train_T
    self.sampling_T = sampling_T or train_T  # default: full schedule

    betas = torch.linspace(beta_1, beta_T, train_T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    indices = torch.linspace(self.train_T - 1, 0, self.sampling_T, dtype=torch.long)
    self.sampling_indices = indices.tolist()

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

      beta_t = extract(self.betas, t_vec, x.shape)
      sqrt_beta_t = extract(self.sqrt_betas, t_vec, x.shape)
      sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t_vec, x.shape)
      sqrt_recip_alpha_t = extract(self.sqrt_recip_alphas, t_vec, x.shape)

      x = sqrt_recip_alpha_t * (x - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)

      if t > 0:
        noise = torch.randn_like(x)
        x += sqrt_beta_t * noise

    return x

class DDIMSampler(nn.Module):
  def __init__(self, model, beta_1, beta_T, train_T, sampling_T=None, eta=0.0):
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
    dir_coeff = torch.zeros_like(alphas_bar)

    if eta > 0:
      frac = (1 - alphas_bar_prev) / (1 - alphas_bar)
      inner = 1 - (alphas_bar / alphas_bar_prev)
      sigma = eta * torch.sqrt(torch.clamp(frac * inner, min=0.0))

    dir_coeff = torch.sqrt(torch.clamp(1 - alphas_bar_prev - sigma**2, min=0.0))

    step = train_T // self.sampling_T
    self.sampling_indices = reversed(torch.arange(0, train_T, step)[:self.sampling_T].tolist())

    self.register_buffer("alphas", alphas)
    self.register_buffer("alphas_bar", alphas_bar)
    self.register_buffer("alphas_bar_prev", alphas_bar_prev)
    self.register_buffer("sqrt_alphas_bar", sqrt_alphas_bar)
    self.register_buffer("sqrt_one_minus_alphas_bar", sqrt_one_minus_alphas_bar)
    self.register_buffer("sqrt_alphas_bar_prev", sqrt_alphas_bar_prev)
    self.register_buffer("sigma", sigma)
    self.register_buffer("dir_coeff", dir_coeff)

  def forward(self, x_T, condition):
    x = x_T

    for t in self.sampling_indices:
      t_tensor = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
      eps = self.model(x, condition, t_tensor)

      sqrt_alpha_bar_t = extract(self.sqrt_alphas_bar, t_tensor, x.shape)
      sqrt_one_minus_alpha_bar_t = extract(self.sqrt_one_minus_alphas_bar, t_tensor, x.shape)
      sqrt_a_bar_prev = extract(self.sqrt_alphas_bar_prev, t_tensor, x.shape)
      sigma = extract(self.sigma, t_tensor, x.shape)
      dir_coeff = extract(self.dir_coeff, t_tensor, x.shape)

      x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / sqrt_alpha_bar_t

      noise = sigma * torch.randn_like(x) if t > 0 else 0
      x = sqrt_a_bar_prev * x0 + dir_coeff * eps + noise

    return x
