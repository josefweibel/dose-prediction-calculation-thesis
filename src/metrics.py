import torch
from torchmetrics import Metric

class DoseScore(Metric):
  def __init__(self, compute_on_step=True, dist_sync_on_step=False):
    super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
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
    score = (preds - target) / target
    self.total += score.sum()
    self.count += target.numel()

  def compute(self):
    # average over all elements
    return self.total / self.count

class DVHScore(Metric):
  def __init__(self, compute_on_step=True, dist_sync_on_step=False):
    super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def _percentile(self, tensor: torch.Tensor, q: float) -> torch.Tensor:
    """Compute percentile for a flattened tensor."""
    k = 1 + round(0.01 * q * (tensor.numel() - 1))
    return tensor.kthvalue(k).values

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    preds: predicted doses, shape (batch_size, ...)
    target: actual doses, shape (batch_size, ...)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      p = preds[i].flatten()
      t = target[i].flatten()

      D1_hat = self._percentile(p, 1)
      D95_hat = self._percentile(p, 95)
      D99_hat = self._percentile(p, 99)

      D1 = self._percentile(t, 1)
      D95 = self._percentile(t, 95)
      D99 = self._percentile(t, 99)

      score_i = (torch.abs(D1_hat - D1) +
                  torch.abs(D95_hat - D95) +
                  torch.abs(D99_hat - D99)) / 3.0

      self.total += score_i
      self.count += 1

  def compute(self):
    return self.total / self.count

class HomogeneityIndex(Metric):
  def __init__(self, compute_on_step=True, dist_sync_on_step=False):
    super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def _percentile(self, tensor: torch.Tensor, q: float) -> torch.Tensor:
    """Compute percentile for a flattened tensor."""
    k = 1 + round(0.01 * q * (tensor.numel() - 1))
    return tensor.kthvalue(k).values

  def update(self, dose: torch.Tensor):
    """
    dose: dose distribution, shape (batch_size, ...)
    """
    batch_size = dose.shape[0]

    for i in range(batch_size):
      d = dose[i].flatten()

      D2 = self._percentile(d, 2)
      D98 = self._percentile(d, 98)
      D50 = self._percentile(d, 50)

      hi_value = (D2 - D98) / D50
      self.total += hi_value
      self.count += 1

  def compute(self):
    return self.total / self.count
