import torch
from torchmetrics import Metric

__all__ = [
  'DoseScore',
  'DVHScore',
  'DVHAUC',
  'HomogeneityIndex',
  'BodyMaskedMetric',
  'ChallengeMAE',
  'VarianceOfLaplacian',
  'RegionMaskedMetric',
  'PTVPrescriptionMAE',
  'PTVPrescriptionMeanDose',
  'GammaIndex',
]


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


class DVHAUC(Metric):
  """
  Area Under the Curve (AUC) for Dose-Volume Histogram (DVH).

  This metric computes the AUC of the DVH curve, which represents the integral
  of the volume percentage as a function of dose. A higher AUC indicates that
  more volume receives higher doses.

  The DVH curve shows what percentage of the volume receives at least a certain dose.
  The AUC is computed using the trapezoidal rule for numerical integration.

  Works with both 2D slices and 3D volumes.

  Args:
    num_bins: Number of dose bins for DVH computation (default: 100)
    normalize: If True, normalize AUC by the maximum possible area (default: True)
    dose_range: Tuple of (min_dose, max_dose) to use for binning. If None, uses per-sample min/max (default: None)
  """
  def __init__(self, num_bins: int = 100, normalize: bool = True, dose_range: tuple = None, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.num_bins = num_bins
    self.normalize = normalize
    self.dose_range = dose_range  # (min_dose, max_dose) or None

    self.add_state("total_auc_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_auc_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_auc_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def _compute_dvh_auc(self, dose_volume: torch.Tensor) -> torch.Tensor:
    """
    Compute AUC for a single dose volume.

    Args:
      dose_volume: Dose values, shape (...) - can be any shape, will be flattened

    Returns:
      AUC value as scalar tensor
    """
    doses = dose_volume.flatten()

    if doses.numel() == 0:
      return torch.tensor(0.0, device=doses.device)

    # Get min and max dose for binning
    if self.dose_range is not None:
      # Use fixed dose range from dataset
      min_dose = torch.tensor(self.dose_range[0], device=doses.device, dtype=doses.dtype)
      max_dose = torch.tensor(self.dose_range[1], device=doses.device, dtype=doses.dtype)
    else:
      # Use per-sample min/max
      min_dose = doses.min()
      max_dose = doses.max()

    # Handle case where all doses are the same (only relevant for per-sample mode)
    if max_dose - min_dose < 1e-10:
      # If all doses are the same, DVH is a step function
      # AUC = dose_value * 1.0 (100% volume receives this dose)
      return max_dose if not self.normalize else torch.tensor(1.0, device=doses.device)

    # Create dose bins
    dose_bins = torch.linspace(min_dose, max_dose, self.num_bins, device=doses.device)

    # Compute cumulative DVH: for each dose level, what % of volume receives >= that dose
    dvh_values = torch.zeros(self.num_bins, device=doses.device)
    for i, dose_level in enumerate(dose_bins):
      dvh_values[i] = (doses >= dose_level).float().mean() * 100.0  # Percentage

    # Compute AUC using trapezoidal rule
    # AUC = integral of volume(dose) over dose range
    # Trapezoidal rule: sum of (dose[i+1] - dose[i]) * (volume[i] + volume[i+1]) / 2
    dose_diffs = dose_bins[1:] - dose_bins[:-1]
    volume_avgs = (dvh_values[1:] + dvh_values[:-1]) / 2.0
    auc = torch.sum(dose_diffs * volume_avgs)

    if self.normalize:
      # Normalize by maximum possible area: max_dose * 100 (if 100% volume received max dose)
      max_area = (max_dose - min_dose) * 100.0
      auc = auc / max_area if max_area > 1e-10 else torch.tensor(0.0, device=doses.device)

    return auc

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    Update the metric with new predictions and targets.

    Args:
      preds: predicted doses, shape (batch_size, ...) - works with 2D or 3D
      target: reference doses, shape (batch_size, ...) - works with 2D or 3D
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred_auc = self._compute_dvh_auc(preds[i])
      target_auc = self._compute_dvh_auc(target[i])

      self.total_auc_pred += pred_auc
      self.total_auc_target += target_auc
      self.total_auc_diff += torch.abs(pred_auc - target_auc)
      self.count += 1

  def compute(self):
    """
    Compute the average DVH AUC statistics across all samples.

    Returns:
      Dictionary with pred_auc (avg predicted AUC), target_auc (avg target AUC),
      and auc_diff (avg absolute difference between predicted and target AUC)
    """
    if self.count == 0:
      return {
        'pred': torch.tensor(float('nan')),
        'target': torch.tensor(float('nan')),
        'diff': torch.tensor(float('nan'))
      }

    return {
      'pred': self.total_auc_pred / self.count,
      'target': self.total_auc_target / self.count,
      'diff': self.total_auc_diff / self.count
    }


class GammaIndex(Metric):
  """
  3D Gamma Index metric for comparing dose distributions in radiotherapy.

  The gamma index combines dose difference and distance-to-agreement (DTA) criteria
  to evaluate the similarity between two dose distributions. A gamma value <= 1
  indicates agreement within the specified tolerances.

  The metric computes:
  - Gamma pass rate: percentage of voxels with gamma <= 1
  - Mean gamma: average gamma value across all evaluated voxels
  - Median gamma: median gamma value

  Args:
    dose_tolerance: Dose difference criterion in % (e.g., 3 for 3%)
    distance_tolerance: Distance-to-agreement criterion in mm (e.g., 3 for 3mm)
    dose_threshold: Minimum dose threshold in % of max dose (default: 10%)
                    Only voxels above this threshold are evaluated
    voxel_spacing: Physical spacing between voxels in mm (x, y, z). Default: (2.5, 2.5, 2.0)
    global_normalization: If True, normalize by global max dose. If False, use local dose (default: True)
    search_radius: Maximum search radius in mm for finding minimum gamma (default: 2x distance_tolerance)
  """
  def __init__(
    self,
    dose_tolerance: float = 3.0,
    distance_tolerance: float = 3.0,
    dose_threshold: float = 10.0,
    voxel_spacing: tuple = (2.5, 2.5, 2.0),
    global_normalization: bool = True,
    search_radius: float = None,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)

    self.dose_tolerance = dose_tolerance / 100.0  # Convert percentage to fraction
    self.distance_tolerance = distance_tolerance
    self.dose_threshold = dose_threshold / 100.0  # Convert percentage to fraction
    self.voxel_spacing = torch.tensor(voxel_spacing)
    self.global_normalization = global_normalization
    self.search_radius = search_radius if search_radius is not None else 2 * distance_tolerance

    # Compute search window size in voxels for each dimension
    self.search_window = tuple(
      int(torch.ceil(torch.tensor(self.search_radius / s)).item())
      for s in voxel_spacing
    )

    self.add_state("total_pass_rate", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_mean_gamma", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_median_gamma", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def _compute_gamma_for_point(
    self,
    ref_dose_point: torch.Tensor,
    eval_dose_region: torch.Tensor,
    distances: torch.Tensor,
    normalization_dose: float
  ) -> torch.Tensor:
    """
    Compute gamma value for a single reference point by searching in evaluation dose region.

    Args:
      ref_dose_point: Reference dose at the point (scalar)
      eval_dose_region: Evaluation doses in search region (tensor)
      distances: Physical distances to each point in search region (tensor, same shape as eval_dose_region)
      normalization_dose: Dose value for normalization

    Returns:
      Minimum gamma value found in the search region
    """
    # Dose difference component
    dose_diff = torch.abs(eval_dose_region - ref_dose_point) / (self.dose_tolerance * normalization_dose)

    # Distance component
    distance_component = distances / self.distance_tolerance

    # Gamma calculation: sqrt((dose_diff/tolerance)^2 + (distance/tolerance)^2)
    gamma = torch.sqrt(dose_diff**2 + distance_component**2)

    # Return minimum gamma in the search region
    return torch.min(gamma)

  def update(self, preds: torch.Tensor, target: torch.Tensor):
    """
    Update the gamma index metric with new predictions and targets.

    Args:
      preds: predicted doses, shape (batch_size, 1, H, W, D) or (batch_size, H, W, D)
      target: reference doses, shape (batch_size, 1, H, W, D) or (batch_size, H, W, D)
    """
    # Handle both (B, 1, H, W, D) and (B, H, W, D) shapes
    if preds.dim() == 4:
      preds = preds.unsqueeze(1)
    if target.dim() == 4:
      target = target.unsqueeze(1)

    batch_size = preds.shape[0]
    self.voxel_spacing = self.voxel_spacing.to(preds.device)

    for b in range(batch_size):
      pred_vol = preds[b, 0]  # (H, W, D)
      ref_vol = target[b, 0]  # (H, W, D)

      # Determine normalization dose
      if self.global_normalization:
        normalization_dose = torch.max(ref_vol)
      else:
        normalization_dose = None  # Will use local dose for each point

      # Create mask for voxels above dose threshold
      max_dose = torch.max(ref_vol)
      dose_mask = ref_vol >= (self.dose_threshold * max_dose)

      # Get coordinates of voxels to evaluate
      eval_coords = torch.nonzero(dose_mask, as_tuple=False)  # (N, 3) where N is number of voxels above threshold

      if eval_coords.shape[0] == 0:
        continue

      gamma_values = []

      # For each reference point above threshold
      for coord in eval_coords:
        x, y, z = coord[0].item(), coord[1].item(), coord[2].item()
        ref_dose_point = ref_vol[x, y, z]

        # Use local normalization if specified
        local_norm = ref_dose_point if not self.global_normalization else normalization_dose

        # Define search region bounds
        x_min = max(0, x - self.search_window[0])
        x_max = min(ref_vol.shape[0], x + self.search_window[0] + 1)
        y_min = max(0, y - self.search_window[1])
        y_max = min(ref_vol.shape[1], y + self.search_window[1] + 1)
        z_min = max(0, z - self.search_window[2])
        z_max = min(ref_vol.shape[2], z + self.search_window[2] + 1)

        # Extract evaluation dose region
        eval_dose_region = pred_vol[x_min:x_max, y_min:y_max, z_min:z_max]

        # Compute distances from reference point to all points in search region
        x_coords = torch.arange(x_min, x_max, device=preds.device).float()
        y_coords = torch.arange(y_min, y_max, device=preds.device).float()
        z_coords = torch.arange(z_min, z_max, device=preds.device).float()

        # Create meshgrid for distances
        zz, yy, xx = torch.meshgrid(z_coords, y_coords, x_coords, indexing='ij')

        # Transpose to match (x, y, z) order
        xx = xx.permute(2, 1, 0)
        yy = yy.permute(2, 1, 0)
        zz = zz.permute(2, 1, 0)

        # Calculate physical distances
        dx = (xx - x) * self.voxel_spacing[0]
        dy = (yy - y) * self.voxel_spacing[1]
        dz = (zz - z) * self.voxel_spacing[2]
        distances = torch.sqrt(dx**2 + dy**2 + dz**2)

        # Compute gamma for this reference point
        gamma = self._compute_gamma_for_point(
          ref_dose_point,
          eval_dose_region,
          distances,
          local_norm
        )
        gamma_values.append(gamma)

      # Convert to tensor
      gamma_values = torch.stack(gamma_values)

      # Compute statistics
      pass_rate = (gamma_values <= 1.0).float().mean() * 100.0  # Percentage
      mean_gamma = gamma_values.mean()
      median_gamma = torch.median(gamma_values)

      self.total_pass_rate += pass_rate
      self.total_mean_gamma += mean_gamma
      self.total_median_gamma += median_gamma
      self.count += 1

  def compute(self):
    """
    Compute the average gamma index statistics across all samples.

    Returns:
      Dictionary with pass_rate, mean_gamma, and median_gamma
    """
    if self.count == 0:
      return {
        'pass_rate': torch.tensor(float('nan')),
        'mean_gamma': torch.tensor(float('nan')),
        'median_gamma': torch.tensor(float('nan'))
      }

    return {
      'pass_rate': self.total_pass_rate / self.count,
      'mean_gamma': self.total_mean_gamma / self.count,
      'median_gamma': self.total_median_gamma / self.count
    }
