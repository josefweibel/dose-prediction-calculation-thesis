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
    if self.count == 0:
      return torch.tensor(float('nan'))
    return self.total / self.count

class BodyMaskedMetric(Metric):
  def __init__(self, base_metric: Metric, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.base_metric = base_metric
    self.add_state("samples_processed", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
    mask = mask.to(torch.bool) > 0
    batch_size = preds.shape[0]

    for i in range(batch_size):
      p = preds[i][mask[i]].flatten()
      t = target[i][mask[i]].flatten()

      if p.numel() == 0 or t.numel() == 0:
        continue

      self.base_metric.update(p.unsqueeze(0), t.unsqueeze(0))
      self.samples_processed += 1

  def compute(self):
    if self.samples_processed == 0:
      return torch.tensor(float('nan'))
    return self.base_metric.compute()

  def reset(self):
    """Reset both this metric and the base metric."""
    super().reset()
    self.base_metric.reset()

  def to(self, *args, **kwargs):
    """Move both this metric and the base metric to the specified device/dtype."""
    super().to(*args, **kwargs)
    self.base_metric = self.base_metric.to(*args, **kwargs)
    return self


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
      preds: predicted doses, shape (batch_size, 1, H, W)
      target: reference doses, shape (batch_size, 1, H, W)
      ptv_high_mask: PTV_High structure mask, shape (batch_size, H, W)
      body_mask: Body structure mask, shape (batch_size, H, W)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0]  # Remove channel dimension: (H, W)
      ref_dose = target[i, 0]  # Remove channel dimension: (H, W)
      ptv_mask = ptv_high_mask[i].to(torch.bool)  # (H, W)
      body = body_mask[i].to(torch.bool)  # (H, W)

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


class VarianceOfLaplacian(Metric):
  """
  Variance of Laplacian metric for measuring image sharpness/blurriness.

  Higher values indicate sharper images, while lower values indicate blurrier images.
  This metric applies a Laplacian filter to the image and computes the variance
  of the resulting filtered image.
  """
  def __init__(self, mode: str = "preds", *args, **kwargs):
    """
    Args:
      mode: Which image to compute variance for. Options:
        - "preds": compute variance of Laplacian for predictions
        - "target": compute variance of Laplacian for targets
        - "diff": compute variance of Laplacian for the difference (preds - target)
    """
    super().__init__(*args, **kwargs)
    if mode not in ["preds", "target", "diff"]:
      raise ValueError(f"mode must be 'preds', 'target', or 'diff', got '{mode}'")
    self.mode = mode

    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    Update the metric with new predictions and targets.

    Args:
      preds: predicted images, shape (batch_size, channels, H, W) or (batch_size, H, W)
      target: target images, shape (batch_size, channels, H, W) or (batch_size, H, W)
    """
    # Handle both (B, C, H, W) and (B, H, W) shapes
    if preds.dim() == 3:
      preds = preds.unsqueeze(1)  # Add channel dimension
    if target.dim() == 3:
      target = target.unsqueeze(1)  # Add channel dimension

    batch_size = preds.shape[0]

    # Define 3x3 Laplacian kernel
    # Standard Laplacian: [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
    laplacian_kernel = torch.tensor([[0, 1, 0],
                                      [1, -4, 1],
                                      [0, 1, 0]],
                                     dtype=preds.dtype,
                                     device=preds.device)
    laplacian_kernel = laplacian_kernel.view(1, 1, 3, 3)

    for i in range(batch_size):
      pred_img = preds[i]  # (C, H, W)
      target_img = target[i]  # (C, H, W)

      # Select image based on mode
      if self.mode == "preds":
        img = pred_img
      elif self.mode == "target":
        img = target_img
      else:  # mode == "diff"
        img = pred_img - target_img

      # Apply Laplacian to each channel and average
      channel_vars = []
      for c in range(img.shape[0]):
        channel = img[c:c+1].unsqueeze(0)  # (1, 1, H, W)

        # Apply convolution with Laplacian kernel
        laplacian = torch.nn.functional.conv2d(
          channel,
          laplacian_kernel,
          padding=1
        )

        # Compute variance of the Laplacian
        var = torch.var(laplacian)
        channel_vars.append(var)

      # Average variance across channels
      mean_var = torch.stack(channel_vars).mean()
      self.total += mean_var
      self.count += 1

  def compute(self):
    """Compute the average Variance of Laplacian across all samples."""
    if self.count == 0:
      return torch.tensor(0.0)
    return self.total / self.count


class RegionMaskedMetric(Metric):
  """
  Wrapper metric that applies a base metric to specific anatomical regions (PTVs or OARs).

  This metric takes a region mask and only evaluates the base metric on pixels within that region.
  It automatically handles region indexing from the structure tensor and computes the metric
  only on the masked pixels.

  Example:
    # Create a metric for MSE on PTV region
    ptv_mse = RegionMaskedMetric(MeanSquaredError(), region_name='PTV', region_index=1)

    # Create a metric for PSNR on Brainstem region
    brainstem_psnr = RegionMaskedMetric(
      PeakSignalNoiseRatio(data_range=1.0),
      region_name='BrainStem_03',
      region_index=13
    )
  """
  def __init__(self, base_metric: Metric, region_name: str, region_index: int, *args, **kwargs):
    """
    Args:
      base_metric: The underlying metric to compute (e.g., MeanSquaredError, PSNR, etc.)
      region_name: Name of the region (e.g., 'PTV', 'BrainStem_03')
      region_index: Index of the region in the structure tensor (0=CT, 1=PTV, 2+=OARs)
    """
    super().__init__(*args, **kwargs)
    self.base_metric = base_metric
    self.region_name = region_name
    self.region_index = region_index
    self.add_state("samples_processed", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, structure: torch.Tensor):
    """
    Update the metric with new predictions and targets, masked by the region.

    Args:
      preds: predicted doses, shape (batch_size, 1, H, W)
      target: true doses, shape (batch_size, 1, H, W)
      structure: structure tensor with all regions, shape (batch_size, n_channels, H, W)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      # Extract the region mask for this sample
      region_mask = structure[i, self.region_index].to(torch.bool)

      # Skip if region is empty
      if region_mask.sum() == 0:
        continue

      # Extract predictions and targets within the region
      pred_region = preds[i][region_mask.unsqueeze(0)].flatten()
      target_region = target[i][region_mask.unsqueeze(0)].flatten()

      if pred_region.numel() == 0 or target_region.numel() == 0:
        continue

      # Update the base metric with the masked values
      # Add batch dimension back for compatibility
      self.base_metric.update(pred_region.unsqueeze(0), target_region.unsqueeze(0))
      self.samples_processed += 1

  def compute(self):
    """Compute the metric value from the base metric."""
    if self.samples_processed == 0:
      return torch.tensor(float('nan'))
    return self.base_metric.compute()

  def reset(self):
    """Reset both this metric and the base metric."""
    super().reset()
    self.base_metric.reset()

  def to(self, *args, **kwargs):
    """Move both this metric and the base metric to the specified device/dtype."""
    super().to(*args, **kwargs)
    self.base_metric = self.base_metric.to(*args, **kwargs)
    return self


class PTVPrescriptionMAE(Metric):
  """
  Mean Absolute Error between predicted dose and prescribed dose for a specific PTV level.

  This metric:
  1. Calculates the exclusive region for the specified PTV level:
     - PTV70: ptv70 mask (highest prescription)
     - PTV63: ptv63 - ptv70 (medium prescription, excluding high)
     - PTV56: ptv56 - ptv63 (lowest prescription, excluding medium and high)
  2. Computes MAE between predicted doses and the prescription level in that region

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56)
  """
  def __init__(self, ptv_level: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if ptv_level not in [70, 63, 56]:
      raise ValueError(f"ptv_level must be 70, 63, or 56, got {ptv_level}")

    self.ptv_level = ptv_level
    self.prescribed_dose = float(ptv_level)
    self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, ptv70_mask: torch.Tensor,
             ptv63_mask: torch.Tensor, ptv56_mask: torch.Tensor):
    """
    Update the metric with new predictions.

    Args:
      preds: predicted doses in Gy, shape (batch_size, 1, H, W)
      target: true doses in Gy (not used but kept for consistency), shape (batch_size, 1, H, W)
      ptv70_mask: PTV70 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
      ptv63_mask: PTV63 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
      ptv56_mask: PTV56 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0]  # (H, W)

      # Ensure masks are boolean and 2D
      ptv70 = ptv70_mask[i].squeeze().to(torch.bool) if ptv70_mask[i].dim() > 2 else ptv70_mask[i].to(torch.bool)
      ptv63 = ptv63_mask[i].squeeze().to(torch.bool) if ptv63_mask[i].dim() > 2 else ptv63_mask[i].to(torch.bool)
      ptv56 = ptv56_mask[i].squeeze().to(torch.bool) if ptv56_mask[i].dim() > 2 else ptv56_mask[i].to(torch.bool)

      # Calculate exclusive region for the specified PTV level
      if self.ptv_level == 70:
        region_mask = ptv70
      elif self.ptv_level == 63:
        region_mask = ptv63 & ~ptv70
      else:  # 56
        region_mask = ptv56 & ~ptv63

      if region_mask.sum() == 0:
        continue

      # Get predicted dose values in this region
      pred_doses = pred[region_mask]

      # Calculate mean absolute error from prescribed dose
      error = torch.abs(pred_doses - self.prescribed_dose).mean()

      self.total_error += error
      self.count += 1

  def compute(self):
    """Compute the average MAE for this PTV level."""
    if self.count == 0:
      return torch.tensor(float('nan'))
    return self.total_error / self.count


class PTVPrescriptionMeanDose(Metric):
  """
  Mean predicted dose in the exclusive region for a specific PTV level.

  This metric calculates the mean predicted dose in the exclusive region of a PTV level,
  allowing you to see how close predictions are to the prescription level.

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56)
  """
  def __init__(self, ptv_level: int, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if ptv_level not in [70, 63, 56]:
      raise ValueError(f"ptv_level must be 70, 63, or 56, got {ptv_level}")

    self.ptv_level = ptv_level
    self.prescribed_dose = float(ptv_level)
    self.add_state("total_dose", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, ptv70_mask: torch.Tensor,
             ptv63_mask: torch.Tensor, ptv56_mask: torch.Tensor):
    """
    Update the metric with new predictions.

    Args:
      preds: predicted doses in Gy, shape (batch_size, 1, H, W)
      target: true doses in Gy (not used but kept for consistency), shape (batch_size, 1, H, W)
      ptv70_mask: PTV70 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
      ptv63_mask: PTV63 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
      ptv56_mask: PTV56 binary mask, shape (batch_size, 1, H, W) or (batch_size, H, W)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0]  # (H, W)

      # Ensure masks are boolean and 2D
      ptv70 = ptv70_mask[i].squeeze().to(torch.bool) if ptv70_mask[i].dim() > 2 else ptv70_mask[i].to(torch.bool)
      ptv63 = ptv63_mask[i].squeeze().to(torch.bool) if ptv63_mask[i].dim() > 2 else ptv63_mask[i].to(torch.bool)
      ptv56 = ptv56_mask[i].squeeze().to(torch.bool) if ptv56_mask[i].dim() > 2 else ptv56_mask[i].to(torch.bool)

      # Calculate exclusive region for the specified PTV level
      if self.ptv_level == 70:
        region_mask = ptv70
      elif self.ptv_level == 63:
        region_mask = ptv63 & ~ptv70
      else:  # 56
        region_mask = ptv56 & ~ptv63

      if region_mask.sum() == 0:
        continue

      # Get predicted dose values in this region
      pred_doses = pred[region_mask]

      # Calculate mean dose
      mean_dose = pred_doses.mean()

      self.total_dose += mean_dose
      self.count += 1

  def compute(self):
    """Compute the average mean dose for this PTV level."""
    if self.count == 0:
      return torch.tensor(float('nan'))
    return self.total_dose / self.count
