import torch
from torchmetrics import Metric

__all__ = [
  'DoseScore',
  'DVHScore',
  'DVHAUC',
  'DoseLevelMetricBase',
  'HomogeneityIndex',
  'ConformityIndex',
  'GradientIndex',
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


class DoseLevelMetricBase(Metric):
  """
  Base class for metrics that compute values for specific dose levels.

  This base class provides common functionality for metrics that need to compute
  separately for PTV70, PTV63, and PTV56 dose levels, as well as for the highest
  prescribed dose in each sample.

  Subclasses should implement:
    - _compute_metric_for_sample(pred, target, ptv_masks, ptv_level) -> scalar

  The base class handles:
    - State management for all dose levels (70, 63, 56, highest)
    - Determining the highest dose level per sample
    - Aggregating results across samples
    - Computing final averages

  Args:
    ptv_level: Single dose level (70, 63, 56) or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample (only with compute_all_levels)
  """
  def __init__(self, ptv_level: int = None, compute_all_levels: bool = False,
               include_highest: bool = True, *args, **kwargs):
    super().__init__(*args, **kwargs)

    if ptv_level is not None and ptv_level not in [70, 63, 56]:
      raise ValueError(f"ptv_level must be 70, 63, or 56, got {ptv_level}")

    self.ptv_level = ptv_level
    self.compute_all_levels = compute_all_levels
    self.include_highest = include_highest

    if compute_all_levels:
      # Track sum and count for each dose level
      self.add_state("sum_70", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_70", default=torch.tensor(0), dist_reduce_fx="sum")
      self.add_state("sum_63", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_63", default=torch.tensor(0), dist_reduce_fx="sum")
      self.add_state("sum_56", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_56", default=torch.tensor(0), dist_reduce_fx="sum")

      if include_highest:
        self.add_state("sum_highest", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count_highest", default=torch.tensor(0), dist_reduce_fx="sum")
    else:
      # Single level mode
      self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def _get_region_mask(self, ptv70: torch.Tensor, ptv63: torch.Tensor,
                       ptv56: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Calculate the exclusive region mask for a specific dose level.

    Args:
      ptv70: PTV70 mask for single sample
      ptv63: PTV63 mask for single sample
      ptv56: PTV56 mask for single sample
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Boolean mask for the exclusive region
    """
    if ptv_level == 70:
      return ptv70
    elif ptv_level == 63:
      return ptv63 & ~ptv70
    else:  # 56
      return ptv56 & ~ptv63

  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute the metric for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample, shape (H, W) or (H, W, D)
      target: Target dose for single sample, shape (H, W) or (H, W, D)
      region_mask: Boolean mask for the exclusive PTV region, shape (H, W) or (H, W, D)
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Metric value as a scalar tensor
    """
    raise NotImplementedError("Subclasses must implement _compute_metric_for_sample")

  def update(self, preds: torch.Tensor, target: torch.Tensor,
             ptv70_mask: torch.Tensor, ptv63_mask: torch.Tensor, ptv56_mask: torch.Tensor):
    """Update the metric with new predictions."""
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0] if preds.dim() > 3 else preds[i]
      tgt = target[i, 0] if target.dim() > 3 else target[i]
      ptv70 = ptv70_mask[i].squeeze().to(torch.bool)
      ptv63 = ptv63_mask[i].squeeze().to(torch.bool)
      ptv56 = ptv56_mask[i].squeeze().to(torch.bool)

      if self.compute_all_levels:
        # Determine highest level for this sample
        highest_level = None
        if self.include_highest:
          if ptv70.sum() > 0:
            highest_level = 70
          elif ptv63.sum() > 0:
            highest_level = 63
          elif ptv56.sum() > 0:
            highest_level = 56

        # Compute for all levels and reuse score for highest
        for level in [70, 63, 56]:
          region_mask = self._get_region_mask(ptv70, ptv63, ptv56, level)
          score = self._compute_metric_for_sample(pred, tgt, region_mask, level)
          if not torch.isnan(score):
            if level == 70:
              self.sum_70 += score
              self.count_70 += 1
            elif level == 63:
              self.sum_63 += score
              self.count_63 += 1
            else:  # 56
              self.sum_56 += score
              self.count_56 += 1

            # Reuse this score for highest if it matches
            if level == highest_level:
              self.sum_highest += score
              self.count_highest += 1
      else:
        # Single level mode
        region_mask = self._get_region_mask(ptv70, ptv63, ptv56, self.ptv_level)
        score = self._compute_metric_for_sample(pred, tgt, region_mask, self.ptv_level)
        if not torch.isnan(score):
          self.total += score
          self.count += 1

  def compute(self):
    """Compute the final metric value(s)."""
    if self.compute_all_levels:
      results = {}
      if self.count_70 > 0:
        results[70] = self.sum_70 / self.count_70
      else:
        results[70] = torch.tensor(float('nan'))

      if self.count_63 > 0:
        results[63] = self.sum_63 / self.count_63
      else:
        results[63] = torch.tensor(float('nan'))

      if self.count_56 > 0:
        results[56] = self.sum_56 / self.count_56
      else:
        results[56] = torch.tensor(float('nan'))

      if self.include_highest:
        if self.count_highest > 0:
          results['highest'] = self.sum_highest / self.count_highest
        else:
          results['highest'] = torch.tensor(float('nan'))

      return results
    else:
      if self.count > 0:
        return self.total / self.count
      else:
        return torch.tensor(float('nan'))


class HomogeneityIndex(DoseLevelMetricBase):
  """
  Homogeneity Index (HI) for measuring dose uniformity within a PTV.

  The Homogeneity Index is calculated as:
  HI = (D2 - D98) / D50

  Where:
  - D2: Dose received by 2% of the PTV (near-maximum dose)
  - D98: Dose received by 98% of the PTV (near-minimum dose)
  - D50: Median dose

  A lower HI indicates better dose homogeneity (more uniform dose distribution).
  Values closer to 0 are ideal.

  This metric computes HI for a specific PTV level in its exclusive region.

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56), or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample
  """
  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute homogeneity index for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample
      target: Target dose (not used but required by interface)
      region_mask: Boolean mask for the exclusive PTV region
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Homogeneity index value
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get dose values in the region
    doses = pred[region_mask].flatten()
    if doses.numel() == 0:
      return torch.tensor(float('nan'))

    # Compute HI
    D2 = torch.quantile(doses, 1 - 0.02)
    D98 = torch.quantile(doses, 1 - 0.98)
    D50 = torch.quantile(doses, 1 - 0.5)

    if D50 < 1e-10:
      D50 = torch.tensor(1e-10, device=doses.device, dtype=doses.dtype)

    return (D2 - D98) / D50

class ConformityIndex(DoseLevelMetricBase):
  """
  Conformity Index (CI) for evaluating how well the prescribed dose conforms to the PTV.

  The Conformity Index is calculated as:
  CI = V_PTV_Rx / V_PTV

  Where:
  - V_PTV_Rx: Volume of PTV receiving at least the prescribed dose
  - V_PTV: Total PTV volume

  A CI of 1.0 indicates perfect conformity (100% of PTV receives prescribed dose).
  Values < 1.0 indicate underdosing of the target.

  This metric computes CI for a specific PTV level in its exclusive region.

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56), or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample
  """
  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute conformity index for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample
      target: Target dose (not used but required by interface)
      region_mask: Boolean mask for the exclusive PTV region
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Conformity index value
    """
    # Total PTV volume (number of voxels in the exclusive PTV region)
    v_ptv = region_mask.sum().float()

    if v_ptv == 0:
      return torch.tensor(float('nan'))

    # Volume of PTV receiving at least the prescribed dose
    prescribed_dose = float(ptv_level)
    dose_coverage_mask = pred >= prescribed_dose
    v_ptv_rx = (region_mask & dose_coverage_mask).sum().float()

    # Calculate conformity index
    return v_ptv_rx / v_ptv


class GradientIndex(DoseLevelMetricBase):
  """
  Paddick Gradient Index (GI) for measuring dose falloff steepness outside the target.

  The Gradient Index quantifies the dose gradient by computing the ratio of the volume
  receiving half the prescription dose to the volume receiving the prescription dose:

  GI = V(half_prescription_dose) / V(prescription_dose)

  A lower GI indicates a steeper dose falloff (better), while a higher GI indicates
  a more gradual falloff (worse). The GI complements conformity indices by specifically
  measuring how quickly the dose decreases outside the target volume.

  Reference:
    Paddick I, Lippitz B. "A simple dose gradient measurement tool to complement
    the conformity index." J Neurosurg. 2006;105(Suppl):194-201.

  This metric computes GI for a specific PTV level.

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56), or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample

  Works with both 2D slices and 3D volumes.
  """
  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute gradient index for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample
      target: Target dose (not used but required by interface)
      region_mask: Boolean mask for the exclusive PTV region
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Gradient index value
    """
    # Skip if PTV region is empty
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Calculate V_prescription: volume receiving at least the prescribed dose
    prescribed_dose = float(ptv_level)
    prescription_mask = pred >= prescribed_dose
    v_prescription = prescription_mask.sum().float()

    # Skip if no voxels receive the prescription dose
    if v_prescription == 0:
      return torch.tensor(float('nan'))

    # Calculate V_half_prescription: volume receiving at least half the prescribed dose
    half_prescribed_dose = prescribed_dose / 2.0
    half_prescription_mask = pred >= half_prescribed_dose
    v_half_prescription = half_prescription_mask.sum().float()

    # Calculate Gradient Index: GI = V(half_Rx) / V(Rx)
    return v_half_prescription / v_prescription


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


class PTVPrescriptionMAE(DoseLevelMetricBase):
  """
  Mean Absolute Error between predicted dose and prescribed dose for a specific PTV level.

  This metric:
  1. Calculates the exclusive region for the specified PTV level:
     - PTV70: ptv70 mask (highest prescription)
     - PTV63: ptv63 - ptv70 (medium prescription, excluding high)
     - PTV56: ptv56 - ptv63 (lowest prescription, excluding medium and high)
  2. Computes MAE between predicted doses and the prescription level in that region

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56), or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample
  """
  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute MAE from prescription dose for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample
      target: Target dose (not used but required by interface)
      region_mask: Boolean mask for the exclusive PTV region
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      MAE from prescription dose
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get predicted dose values in this region
    pred_doses = pred[region_mask]

    # Calculate mean absolute error from prescribed dose
    prescribed_dose = float(ptv_level)
    return torch.abs(pred_doses - prescribed_dose).mean()


class PTVPrescriptionMeanDose(DoseLevelMetricBase):
  """
  Mean predicted dose in the exclusive region for a specific PTV level.

  This metric calculates the mean predicted dose in the exclusive region of a PTV level,
  allowing you to see how close predictions are to the prescription level.

  Args:
    ptv_level: The PTV prescription level (70, 63, or 56), or None for all levels
    compute_all_levels: If True, compute for all levels (70, 63, 56, highest)
    include_highest: If True, also compute for highest dose per sample
  """
  def _compute_metric_for_sample(self, pred: torch.Tensor, target: torch.Tensor,
                                  region_mask: torch.Tensor, ptv_level: int) -> torch.Tensor:
    """
    Compute mean predicted dose for a single sample at a specific dose level.

    Args:
      pred: Predicted dose for single sample
      target: Target dose (not used but required by interface)
      region_mask: Boolean mask for the exclusive PTV region
      ptv_level: Dose level (70, 63, or 56)

    Returns:
      Mean predicted dose in the region
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get predicted dose values in this region
    pred_doses = pred[region_mask]

    # Calculate mean dose
    return pred_doses.mean()


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
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

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

      # OPTIMIZED: Vectorized gamma computation using local search windows only
      # Count total passes
      total_passes = 0
      total_voxels = eval_coords.shape[0]

      # Extract all reference coordinates and doses
      x_refs = eval_coords[:, 0]  # (N,)
      y_refs = eval_coords[:, 1]  # (N,)
      z_refs = eval_coords[:, 2]  # (N,)
      ref_doses = ref_vol[x_refs, y_refs, z_refs]  # (N,)

      # Pad volume once to handle boundary conditions
      pad_size = max(self.search_window)
      pred_padded = torch.nn.functional.pad(
        pred_vol,
        (pad_size, pad_size, pad_size, pad_size, pad_size, pad_size),
        mode='constant',
        value=0  # Padding with 0 will create large dose differences
      )

      # Pre-compute distance grid once (same for all windows)
      dx_grid = torch.arange(-self.search_window[0], self.search_window[0] + 1, device=preds.device, dtype=torch.float32)
      dy_grid = torch.arange(-self.search_window[1], self.search_window[1] + 1, device=preds.device, dtype=torch.float32)
      dz_grid = torch.arange(-self.search_window[2], self.search_window[2] + 1, device=preds.device, dtype=torch.float32)

      dx, dy, dz = torch.meshgrid(dx_grid, dy_grid, dz_grid, indexing='ij')
      distances = torch.sqrt(
        (dx * self.voxel_spacing[0])**2 +
        (dy * self.voxel_spacing[1])**2 +
        (dz * self.voxel_spacing[2])**2
      )

      # Pre-compute mask for points within search radius
      radius_mask = distances <= self.search_radius

      # Process in batches for memory efficiency
      batch_size_gamma = 50  # Process 50 reference points at a time

      for batch_start in range(0, total_voxels, batch_size_gamma):
        batch_end = min(batch_start + batch_size_gamma, total_voxels)

        # Get batch coordinates
        batch_x = x_refs[batch_start:batch_end]  # (B,)
        batch_y = y_refs[batch_start:batch_end]  # (B,)
        batch_z = z_refs[batch_start:batch_end]  # (B,)
        batch_ref_doses = ref_doses[batch_start:batch_end]  # (B,)
        B = batch_x.shape[0]

        # Determine normalization doses
        if self.global_normalization:
          norm_doses = normalization_dose.expand(B)
        else:
          norm_doses = batch_ref_doses

        batch_gamma_min = []

        for i in range(B):
          # Adjust coordinates for padding
          x_pad = batch_x[i] + pad_size
          y_pad = batch_y[i] + pad_size
          z_pad = batch_z[i] + pad_size

          # Extract search window
          x_start = x_pad - self.search_window[0]
          x_end = x_pad + self.search_window[0] + 1
          y_start = y_pad - self.search_window[1]
          y_end = y_pad + self.search_window[1] + 1
          z_start = z_pad - self.search_window[2]
          z_end = z_pad + self.search_window[2] + 1

          window = pred_padded[x_start:x_end, y_start:y_end, z_start:z_end]  # (Wx, Wy, Wz)

          # Compute gamma for this window (using pre-computed distances)
          dose_diff = torch.abs(window - batch_ref_doses[i]) / (self.dose_tolerance * norm_doses[i])
          distance_component = distances / self.distance_tolerance
          gamma = torch.sqrt(dose_diff**2 + distance_component**2)

          # Only consider points within search radius (using pre-computed mask)
          gamma = torch.where(radius_mask, gamma, torch.tensor(1e10, device=gamma.device))

          # Find minimum gamma
          gamma_min = gamma.min()
          batch_gamma_min.append(gamma_min)

        # Count passes in this batch
        batch_gamma_min = torch.stack(batch_gamma_min)
        total_passes += (batch_gamma_min <= 1.0).float().sum()

      # Compute pass rate percentage
      pass_rate = (total_passes / total_voxels) * 100.0

      self.total_pass_rate += pass_rate
      self.count += 1

  def compute(self):
    """
    Compute the average gamma index pass rate across all samples.

    Returns:
      Dictionary with pass_rate (percentage of voxels with gamma <= 1)
    """
    if self.count == 0:
      return {
        'pass_rate': torch.tensor(float('nan'))
      }

    return {
      'pass_rate': self.total_pass_rate / self.count
    }
