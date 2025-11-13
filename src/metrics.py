import torch
from torchmetrics import Metric


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
    mask = mask.to(torch.bool) > 0
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
    mask = mask.to(torch.bool) > 0
    batch_size = preds.shape[0]

    for i in range(batch_size):
      p = preds[i][mask[i]].flatten()
      t = target[i][mask[i]].flatten()

      if p.numel() == 0 or t.numel() == 0:
        continue

      self.base_metric.update(p.unsqueeze(0), t.unsqueeze(0))

  def compute(self):
    return self.base_metric.compute()


class ChallengeMAE(Metric):
  """
  MAE metric following the exact implementation from:
  https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge/blob/main/evaluation.ipynb

  Key features:
  1. D97 normalization: normalizes dose using the 3rd percentile (D97) of the PTV_High structure
  2. 5 Gy isodose masking: only computes MAE in regions where either reference or prediction > 5 Gy
  3. Body masking: only considers voxels within the body contour
  """
  def __init__(self, prescribed_dose: float = 70.0, *args, **kwargs):
    """
    Args:
      prescribed_dose: The prescribed dose to the PTV_High structure (default: 70 Gy)
    """
    super().__init__(*args, **kwargs)
    self.prescribed_dose = prescribed_dose
    self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, ptv_high_mask: torch.Tensor, body_mask: torch.Tensor):
    """
    Update the metric with new predictions and targets.

    Args:
      preds: predicted doses, shape (batch_size, H, W, D)
      target: reference doses, shape (batch_size, H, W, D)
      ptv_high_mask: PTV_High structure mask, shape (batch_size, H, W, D)
      body_mask: Body structure mask, shape (batch_size, H, W, D)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i]
      ref_dose = target[i]
      ptv_mask = ptv_high_mask[i].to(torch.bool)
      body = body_mask[i].to(torch.bool)

      # Step 1: D97 normalization
      # Extract dose values within PTV_High
      ptv_doses = ref_dose[ptv_mask]

      if ptv_doses.numel() == 0:
        continue

      # Compute D97 (3rd percentile of dose within PTV_High)
      d97 = torch.quantile(ptv_doses, 1 - 0.97)

      # Compute normalization scale: prescribed_dose / D97
      norm_scale = self.prescribed_dose / (d97 + 1e-5)

      # Apply normalization to both reference and prediction
      ref_dose_normalized = ref_dose * norm_scale
      pred_normalized = pred * norm_scale

      # Step 2: Create 5 Gy isodose mask
      # Mask includes regions where either reference or prediction exceeds 5 Gy, within body
      isodose_5gy_mask = ((ref_dose_normalized > 5) | (pred_normalized > 5)) & body

      # Step 3: Compute MAE within the masked region
      if isodose_5gy_mask.sum() == 0:
        continue

      diff = torch.abs(ref_dose_normalized - pred_normalized)
      mae = diff[isodose_5gy_mask].mean()

      self.total_error += mae
      self.count += 1

  def compute(self):
    """Compute the average MAE across all samples."""
    if self.count == 0:
      return torch.tensor(0.0)
    return self.total_error / self.count
