import torch
from torch import nn
from torch.nn import functional as F


def extract(v, t, x_shape):
  """
  Extract some coefficients at specified timesteps, then reshape to
  [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
  """
  out = v.to(t.device).gather(0, t).float()
  return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

class GaussianDiffusionSampler(nn.Module):
  def __init__(self, model, beta_1, beta_T, T):
    super().__init__()

    self.model = model
    self.T = T

    self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
    alphas = 1. - self.betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    alphas_bar_prev = F.pad(alphas_bar, [1, 0], value=1)[:T]

    # calculations for diffusion q(x_t | x_{t-1}) and others
    self.register_buffer('sqrt_recip_alphas_bar', torch.sqrt(1. / alphas_bar))
    self.register_buffer('sqrt_recipm1_alphas_bar', torch.sqrt(1. / alphas_bar - 1))

    # calculations for posterior q(x_{t-1} | x_t, x_0)
    self.register_buffer(
      'posterior_var',
      self.betas * (1. - alphas_bar_prev) / (1. - alphas_bar))
    # below: log calculation clipped because the posterior variance is 0 at
    # the beginning of the diffusion chain
    self.register_buffer(
      'posterior_log_var_clipped',
      # torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]])))
      torch.log(self.posterior_var.clamp(min=1e-20)))
    self.register_buffer(
      'posterior_mean_coef1',
      torch.sqrt(alphas_bar_prev) * self.betas / (1. - alphas_bar))
    self.register_buffer(
      'posterior_mean_coef2',
      torch.sqrt(alphas) * (1. - alphas_bar_prev) / (1. - alphas_bar))

  def q_mean_variance(self, x_0, x_t, t):
    """
    Compute the mean and variance of the diffusion posterior
    q(x_{t-1} | x_t, x_0)
    """
    assert x_0.shape == x_t.shape
    posterior_mean = (
      extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
      extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
    )
    posterior_log_var_clipped = extract(self.posterior_log_var_clipped, t, x_t.shape)
    return posterior_mean, posterior_log_var_clipped

  def predict_xstart_from_eps(self, x_t, t, eps):
    assert x_t.shape == eps.shape
    return (
      extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
      extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
    )

  def p_mean_variance(self, x_t, condition, t):
    eps = self.model(x_t, condition, t)
    x_0 = self.predict_xstart_from_eps(x_t, t, eps)

    # x_0 = torch.clamp(x_0, 0., 1.)
    model_mean, model_log_var = self.q_mean_variance(x_0, x_t, t)

    return model_mean, model_log_var

  def forward(self, x_T, condition):
    x = x_T

    for t in reversed(range(self.T)):
      t_vec = torch.full((x.shape[0],), t, dtype=torch.long, device=x_T.device)
      mean, log_var = self.p_mean_variance(x, condition, t_vec)

      # no noise when t == 0
      if t > 0:
        noise = torch.randn_like(x)
      else:
        noise = 0
      x = mean + torch.exp(0.5 * log_var) * noise

    return x
