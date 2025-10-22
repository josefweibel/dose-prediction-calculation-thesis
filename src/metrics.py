import torch
from torchmetrics import Metric
import numpy as np

class DoseScore(Metric):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    # total score sum and count for averaging
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    Update the state with new predictions and targets.

    preds: predicted doses, shape (batch_size, ...)
    target: true doses, shape (batch_size, ...)
    """
    # flatten tensors to ensure element-wise computation
    preds = preds.flatten()
    target = target.flatten()

    # compute relative differences
    score = torch.abs(preds - target)  # / (target + 1e-5)
    self.total += score.sum()
    self.count += target.numel()

  def compute(self):
    # average over all elements
    return self.total / self.count

class DVHScore(Metric):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    preds: predicted doses, shape (batch_size, ...)
    target: actual doses, shape (batch_size, ...)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      p = preds[i].flatten()
      t = target[i].flatten()

      D1_hat = torch.quantile(p, 1 - 0.01)
      D95_hat = torch.quantile(p, 1 - 0.95)
      D99_hat = torch.quantile(p, 1 - 0.99)

      D1 = torch.quantile(t, 1 - 0.01)
      D95 = torch.quantile(t, 1 - 0.95)
      D99 = torch.quantile(t, 1 - 0.99)

      score_i = (torch.abs(D1_hat - D1) +
                  torch.abs(D95_hat - D95) +
                  torch.abs(D99_hat - D99)) / 3.0

      self.total += score_i
      self.count += 1

  def compute(self):
    return self.total / self.count

class HomogeneityIndex(Metric):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(torch.bool)
    batch_size = preds.shape[0]

    for i in range(batch_size):
      d = preds[i][mask[i:i+1]].flatten()

      if d.numel() == 0:
        continue

      D2 = torch.quantile(d, 1 - 0.02)
      D98 = torch.quantile(d, 1 - 0.98)
      D50 = torch.quantile(d, 1 - 0.5)

      if D50 < 1e-10:
        D50 = 1e-10

      hi_value = (D2 - D98) / D50
      self.total += hi_value
      self.count += 1

  def compute(self):
    return self.total / self.count

class BodyMaskedMetric(Metric):
  def __init__(self, base_metric: Metric, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.base_metric = base_metric

  def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(torch.bool)
    batch_size = preds.shape[0]

    for i in range(batch_size):
      p = preds[i][mask[i:i+1]].flatten()
      t = target[i][mask[i:i+1]].flatten()

      if p.numel() == 0 or t.numel() == 0:
        continue

      self.base_metric.update(p.unsqueeze(0), t.unsqueeze(0))

  def compute(self):
    return self.base_metric.compute()
