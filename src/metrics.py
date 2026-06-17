import math
import torch
from torchmetrics import Metric
import numpy as np
from typing import Dict, Optional

from data import GDPHMM_DOSE_MAX, GDPHMM_OAR_V3
from constraints import (
    ConstraintEvaluator, ConstraintLibrary,
    FractionationType, ConstraintGrade,
    GDPHMM_TO_CONSTRAINT_ORGAN
)

__all__ = [
  'DoseScore',
  'DVHScore',
  'OpenKBPDVHScore',
  'DVHAUC',
  'DoseLevelMetricBase',
  'HomogeneityIndex',
  'ConformityIndex',
  'PaddickConformityIndex',
  'GradientIndex',
  'BodyMaskedMetric',
  'ChallengeMAE',
  'VarianceOfLaplacian',
  'RegionMaskedMetric',
  'PTVPrescriptionMAE',
  'PTVPrescriptionMeanDose',
  'GammaIndex',
  'ConstraintSatisfactionRate',
  'ConstraintAgreementScore',
  'PerOrganConstraintCompliance',
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


class OpenKBPDVHScore(Metric):
  """
  OpenKBP-style DVH (Dose-Volume Histogram) Score.

  For each patient, computes the mean absolute error between predicted and
  reference DVH metrics over all available (structure, metric) pairs:
    - For each PTV level present (high, mid, low):
        D_1, D_95, D_99 - dose received by 1%, 95%, 99% of the structure's volume.
        Uses cumulative masks (a lower level includes the contours of higher levels).
    - For each OAR present:
        D_0.1cc - dose to the hottest 0.1 cm^3 of the structure.
        D_mean  - mean dose over the structure.

  The final value is the mean of patient scores across all evaluated samples.

  Args:
    oar_names: List of OAR names to evaluate. Default: GDPHMM_OAR_V3.
    voxel_spacing: Physical voxel spacing in mm (x, y, z), used to convert
                   D_0.1cc into a voxel count. Default: (2.5, 2.5, 2.0).
  """

  PTV_LEVELS = ['high', 'mid', 'low']
  # Volume fractions (top-X of volume) for PTV DVH metrics: D_1, D_95, D_99.
  PTV_VOLUME_FRACTIONS = [0.01, 0.95, 0.99]
  # OAR target volume (in cm^3) for the near-maximum dose metric.
  OAR_NEAR_MAX_VOLUME_CC = 0.1

  def __init__(
    self,
    oar_names: Optional[list] = None,
    voxel_spacing: tuple = (2.5, 2.5, 2.0),
    *args, **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.oar_names = list(oar_names) if oar_names is not None else list(GDPHMM_OAR_V3)
    self.voxel_volume_cc = float(np.prod(voxel_spacing)) / 1000.0
    self.add_state("total", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  @staticmethod
  def _dose_at_volume_fraction(doses: torch.Tensor, fraction: float) -> torch.Tensor:
    """Dose D such that `fraction` of the voxels in `doses` receive at least D."""
    n = doses.numel()
    k = max(1, min(n, int(math.ceil(n * fraction))))
    # k-th largest value = (n - k + 1)-th smallest
    return torch.kthvalue(doses, n - k + 1).values

  def _dose_at_volume_cc(self, doses: torch.Tensor, volume_cc: float) -> torch.Tensor:
    """Dose D such that `volume_cc` cm^3 of the structure receives at least D."""
    n = doses.numel()
    k = max(1, int(math.ceil(volume_cc / self.voxel_volume_cc)))
    # If the structure is smaller than the requested volume, fall back to its minimum.
    k = min(k, n)
    return torch.kthvalue(doses, n - k + 1).values

  def _ptv_errors(self, pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> list:
    pred_voxels = pred[mask]
    target_voxels = target[mask]
    errors = []
    for fraction in self.PTV_VOLUME_FRACTIONS:
      p = self._dose_at_volume_fraction(pred_voxels, fraction)
      t = self._dose_at_volume_fraction(target_voxels, fraction)
      errors.append(torch.abs(p - t))
    return errors

  def _oar_errors(self, pred: torch.Tensor, target: torch.Tensor,
                  mask: torch.Tensor) -> list:
    pred_voxels = pred[mask]
    target_voxels = target[mask]
    d_01cc_p = self._dose_at_volume_cc(pred_voxels, self.OAR_NEAR_MAX_VOLUME_CC)
    d_01cc_t = self._dose_at_volume_cc(target_voxels, self.OAR_NEAR_MAX_VOLUME_CC)
    d_mean_p = pred_voxels.mean()
    d_mean_t = target_voxels.mean()
    return [torch.abs(d_01cc_p - d_01cc_t), torch.abs(d_mean_p - d_mean_t)]

  def update(self, preds: torch.Tensor, target: torch.Tensor,
             ptv_doses: torch.Tensor, organ_masks: Dict[str, torch.Tensor]):
    """
    Args:
      preds: Predicted doses in Gy, shape (B, 1, H, W) or (B, 1, H, W, D).
      target: Reference doses in Gy, shape (B, 1, H, W) or (B, 1, H, W, D).
      ptv_doses: PTV channels, shape (B, 3, H, W) or (B, 3, H, W, D).
                 Channel order: 0=high, 1=mid, 2=low. Non-zero indicates the
                 PTV's physical contour for that prescription level.
      organ_masks: Dict mapping OAR name to its binary mask, each shape
                   (B, H, W) or (B, H, W, D).
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0] if preds.dim() > 3 else preds[i]
      tgt = target[i, 0] if target.dim() > 3 else target[i]
      sample_ptv = ptv_doses[i]

      patient_errors = []

      # PTVs - cumulative masks (level includes all higher levels' contours).
      cumulative = torch.zeros_like(sample_ptv[0], dtype=torch.bool)
      for level_idx in range(sample_ptv.shape[0]):
        level_mask = sample_ptv[level_idx] > 0
        cumulative = cumulative | level_mask
        if level_mask.sum() == 0:
          # This PTV level doesn't exist for this patient - skip.
          continue
        patient_errors.extend(self._ptv_errors(pred, tgt, cumulative))

      # OARs - D_0.1cc and D_mean.
      for oar_name in self.oar_names:
        if oar_name not in organ_masks:
          continue
        oar_mask = organ_masks[oar_name][i].to(torch.bool)
        if oar_mask.sum() == 0:
          continue
        patient_errors.extend(self._oar_errors(pred, tgt, oar_mask))

      if len(patient_errors) == 0:
        continue

      patient_score = torch.stack(patient_errors).mean()
      self.total = self.total + patient_score
      self.count = self.count + 1

  def compute(self):
    if self.count == 0:
      return torch.tensor(float('nan'))
    return self.total / self.count


class DoseLevelMetricBase(Metric):
  """
  Base class for metrics that compute values for specific PTV dose levels.

  This base class provides common functionality for metrics that need to compute
  separately for PTV high, mid, and low dose levels, using cumulative masks
  (low includes mid and high regions, mid includes high region).

  Subclasses should implement:
    - _compute_metric_for_sample(dose, region_mask, level) -> scalar

  The base class handles:
    - State management for all dose levels (high, mid, low)
    - Creating cumulative masks from ptv_doses
    - Aggregating results across samples
    - Computing final averages

  Args:
    ptv_level: Single dose level ('high', 'mid', 'low') or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
  """
  LEVELS = ['high', 'mid', 'low']

  def __init__(self, ptv_level: str = None, compute_all_levels: bool = False, *args, **kwargs):
    super().__init__(*args, **kwargs)

    if ptv_level is not None and ptv_level not in self.LEVELS:
      raise ValueError(f"ptv_level must be one of {self.LEVELS}, got {ptv_level}")

    self.ptv_level = ptv_level
    self.compute_all_levels = compute_all_levels

    if compute_all_levels:
      # Track sum and count for each dose level (pred, target, and abs_diff separately)
      for level in self.LEVELS:
        self.add_state(f"sum_pred_{level}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(f"sum_target_{level}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(f"sum_abs_diff_{level}", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state(f"count_{level}", default=torch.tensor(0), dist_reduce_fx="sum")
    else:
      # Single level mode
      self.add_state("total_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("total_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("total_abs_diff", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  # Whether to use cumulative masks (True) or exclusive masks (False)
  # Subclasses can override this to use exclusive masks
  use_cumulative_mask = True

  LEVEL_TO_INDEX = {'high': 0, 'mid': 1, 'low': 2}

  def _get_cumulative_mask(self, ptv_doses: torch.Tensor, level: str) -> torch.Tensor:
    """
    Calculate the cumulative region mask for a specific dose level.

    Cumulative masks include higher dose regions:
    - 'high': Just the high dose region (index 0)
    - 'mid': High + mid regions (indices 0-1)
    - 'low': All regions (indices 0-2)

    Args:
      ptv_doses: PTV doses for single sample, shape (3, H, W) or (3, H, W, D)
                 Index 0 = highest dose, 1 = mid, 2 = low
      level: Dose level ('high', 'mid', or 'low')

    Returns:
      Boolean mask for the cumulative region
    """
    if level == 'high':
      return ptv_doses[0] > 0
    elif level == 'mid':
      return (ptv_doses[0] > 0) | (ptv_doses[1] > 0)
    else:  # low
      return (ptv_doses[0] > 0) | (ptv_doses[1] > 0) | (ptv_doses[2] > 0)

  def _get_exclusive_mask(self, ptv_doses: torch.Tensor, level: str) -> torch.Tensor:
    """
    Calculate the exclusive region mask for a specific dose level.

    Exclusive masks only include the specific level's region:
    - 'high': Just the high dose region (index 0)
    - 'mid': Just the mid dose region (index 1)
    - 'low': Just the low dose region (index 2)

    Args:
      ptv_doses: PTV doses for single sample, shape (3, H, W) or (3, H, W, D)
                 Index 0 = highest dose, 1 = mid, 2 = low
      level: Dose level ('high', 'mid', or 'low')

    Returns:
      Boolean mask for the exclusive region
    """
    level_idx = self.LEVEL_TO_INDEX[level]
    return ptv_doses[level_idx] > 0

  def _get_region_mask(self, ptv_doses: torch.Tensor, level: str) -> torch.Tensor:
    """
    Get the region mask based on the use_cumulative_mask flag.

    Args:
      ptv_doses: PTV doses for single sample, shape (3, H, W) or (3, H, W, D)
      level: Dose level ('high', 'mid', or 'low')

    Returns:
      Boolean mask for the region (cumulative or exclusive based on class setting)
    """
    if self.use_cumulative_mask:
      return self._get_cumulative_mask(ptv_doses, level)
    else:
      return self._get_exclusive_mask(ptv_doses, level)

  def _compute_metric_for_sample(self, dose: torch.Tensor, region_mask: torch.Tensor,
                                  prescribed_dose: float) -> torch.Tensor:
    """
    Compute the metric for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample, shape (H, W) or (H, W, D)
      region_mask: Boolean mask for the cumulative PTV region, shape (H, W) or (H, W, D)
      prescribed_dose: The prescribed dose value in Gy (non-normalized)

    Returns:
      Metric value as a scalar tensor
    """
    raise NotImplementedError("Subclasses must implement _compute_metric_for_sample")

  def _get_prescribed_dose(self, ptv_doses: torch.Tensor, level: str) -> float:
    """
    Get the prescribed dose value for a specific level.

    Args:
      ptv_doses: PTV doses for the sample, shape (3, H, W) or (3, H, W, D)
                 Values are normalized (dose / GDPHMM_DOSE_MAX)
      level: Dose level ('high', 'mid', or 'low')

    Returns:
      Prescribed dose value in Gy (non-normalized)
    """
    level_idx = self.LEVEL_TO_INDEX[level]
    normalized_dose = ptv_doses[level_idx].max().item()
    return normalized_dose * GDPHMM_DOSE_MAX

  def update(self, preds: torch.Tensor, target: torch.Tensor, ptv_doses: torch.Tensor):
    """
    Update the metric with new predictions.

    Args:
      preds: Predicted doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
      target: Target doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
      ptv_doses: PTV doses, shape (batch_size, 3, H, W) or (batch_size, 3, H, W, D)
                 Index 0 = highest dose, 1 = mid, 2 = low
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0] if preds.dim() > 3 else preds[i]
      tgt = target[i, 0] if target.dim() > 3 else target[i]
      sample_ptv_doses = ptv_doses[i]  # (3, H, W) or (3, H, W, D)

      if self.compute_all_levels:
        # Compute for all levels
        for level in self.LEVELS:
          region_mask = self._get_region_mask(sample_ptv_doses, level)
          prescribed_dose = self._get_prescribed_dose(sample_ptv_doses, level)
          pred_score = self._compute_metric_for_sample(pred, region_mask, prescribed_dose)
          target_score = self._compute_metric_for_sample(tgt, region_mask, prescribed_dose)
          if not torch.isnan(pred_score) and not torch.isnan(target_score):
            abs_diff = torch.abs(pred_score - target_score)
            setattr(self, f"sum_pred_{level}", getattr(self, f"sum_pred_{level}") + pred_score)
            setattr(self, f"sum_target_{level}", getattr(self, f"sum_target_{level}") + target_score)
            setattr(self, f"sum_abs_diff_{level}", getattr(self, f"sum_abs_diff_{level}") + abs_diff)
            setattr(self, f"count_{level}", getattr(self, f"count_{level}") + 1)
      else:
        # Single level mode
        region_mask = self._get_region_mask(sample_ptv_doses, self.ptv_level)
        prescribed_dose = self._get_prescribed_dose(sample_ptv_doses, self.ptv_level)
        pred_score = self._compute_metric_for_sample(pred, region_mask, prescribed_dose)
        target_score = self._compute_metric_for_sample(tgt, region_mask, prescribed_dose)
        if not torch.isnan(pred_score) and not torch.isnan(target_score):
          self.total_pred += pred_score
          self.total_target += target_score
          self.total_abs_diff += torch.abs(pred_score - target_score)
          self.count += 1

  def compute(self):
    """Compute the final metric value(s)."""
    if self.compute_all_levels:
      results = {}
      for level in self.LEVELS:
        count = getattr(self, f"count_{level}")
        if count > 0:
          results[level] = {
            'pred': getattr(self, f"sum_pred_{level}") / count,
            'target': getattr(self, f"sum_target_{level}") / count,
            'abs_diff': getattr(self, f"sum_abs_diff_{level}") / count
          }
        else:
          results[level] = {
            'pred': torch.tensor(float('nan')),
            'target': torch.tensor(float('nan')),
            'abs_diff': torch.tensor(float('nan'))
          }
      return results
    else:
      if self.count > 0:
        return {
          'pred': self.total_pred / self.count,
          'target': self.total_target / self.count,
          'abs_diff': self.total_abs_diff / self.count
        }
      else:
        return {
          'pred': torch.tensor(float('nan')),
          'target': torch.tensor(float('nan')),
          'abs_diff': torch.tensor(float('nan'))
        }


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

  This metric computes HI for a specific PTV level using cumulative masks.

  Args:
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
  """
  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute homogeneity index for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the cumulative PTV region
      prescribed_dose: Prescribed dose value in Gy (unused for HI calculation)

    Returns:
      Homogeneity index value
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get dose values in the region
    doses = dose[region_mask].flatten()
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
  - V_PTV_Rx: Volume of PTV receiving at least a fraction of the prescribed dose
  - V_PTV: Total PTV volume

  A CI of 1.0 indicates perfect conformity (100% of PTV receives the specified dose threshold).
  Values < 1.0 indicate underdosing of the target.

  This metric computes CI for a specific PTV level using cumulative masks.

  Args:
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
    dose_fraction: Fraction of prescribed dose to use as threshold (default: 1.0 = 100%)
                   For example, 0.95 means 95% of the prescribed dose, 1.0 means 100%
  """
  def __init__(self, ptv_level: str = None, compute_all_levels: bool = False,
               dose_fraction: float = 1.0, *args, **kwargs):
    super().__init__(ptv_level, compute_all_levels, *args, **kwargs)
    self.dose_fraction = dose_fraction

  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute conformity index for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the cumulative PTV region
      prescribed_dose: Prescribed dose value in Gy

    Returns:
      Conformity index value
    """
    # Total PTV volume (number of voxels in the PTV region)
    v_ptv = region_mask.sum().float()

    if v_ptv == 0:
      return torch.tensor(float('nan'))

    # Volume of PTV receiving at least the specified fraction of prescribed dose
    dose_threshold = prescribed_dose * self.dose_fraction
    dose_coverage_mask = dose >= dose_threshold
    v_ptv_rx = (region_mask & dose_coverage_mask).sum().float()

    # Calculate conformity index
    return v_ptv_rx / v_ptv


class PaddickConformityIndex(DoseLevelMetricBase):
  """
  Paddick Conformity Index (COIN) for evaluating target coverage and dose spillage.

  The Paddick Conformity Index is a sophisticated measure that accounts for both
  target coverage and irradiation of healthy tissue:

  COIN = (TV_PIV / TV) × (TV_PIV / PIV)

  Where:
  - TV = Target Volume (PTV volume)
  - PIV = Prescription Isodose Volume (volume receiving at least prescription dose)
  - TV_PIV = Target volume covered by prescription isodose (intersection of TV and PIV)

  The first term (TV_PIV / TV) represents target coverage.
  The second term (TV_PIV / PIV) penalizes dose spillage outside the target.

  A COIN of 1.0 indicates perfect conformity (prescription dose covers exactly the target,
  no more, no less). Values < 1.0 indicate either incomplete target coverage or excessive
  spillage to surrounding tissue.

  Reference:
    Paddick I. "A simple scoring ratio to index the conformity of radiosurgical
    treatment plans." J Neurosurg. 2000;93(Suppl 3):219-222.

  Args:
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
    dose_fraction: Fraction of prescribed dose to use as threshold (default: 1.0 = 100%)

  Works with both 2D slices and 3D volumes.
  """
  def __init__(self, ptv_level: str = None, compute_all_levels: bool = False,
               dose_fraction: float = 1.0, *args, **kwargs):
    super().__init__(ptv_level, compute_all_levels, *args, **kwargs)
    self.dose_fraction = dose_fraction

  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute Paddick Conformity Index for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the cumulative PTV region
      prescribed_dose: Prescribed dose value in Gy

    Returns:
      Paddick Conformity Index value
    """
    # TV: Target Volume (total PTV volume)
    tv = region_mask.sum().float()

    if tv == 0:
      return torch.tensor(float('nan'))

    # PIV: Prescription Isodose Volume (volume receiving at least prescription dose)
    dose_threshold = prescribed_dose * self.dose_fraction
    prescription_mask = dose >= dose_threshold
    piv = prescription_mask.sum().float()

    # Avoid division by zero
    if piv == 0:
      return torch.tensor(0.0)

    # TV_PIV: intersection of target volume and prescription isodose volume
    tv_piv = (region_mask & prescription_mask).sum().float()

    # COIN = (TV_PIV / TV) × (TV_PIV / PIV)
    coverage = tv_piv / tv
    spillage_penalty = tv_piv / piv

    return coverage * spillage_penalty


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
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)

  Works with both 2D slices and 3D volumes.
  """
  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute gradient index for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the cumulative PTV region
      prescribed_dose: Prescribed dose value in Gy

    Returns:
      Gradient index value
    """
    # Skip if PTV region is empty
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Calculate V_prescription: volume receiving at least the prescribed dose
    prescription_mask = dose >= prescribed_dose
    v_prescription = prescription_mask.sum().float()

    # Skip if no voxels receive the prescription dose
    if v_prescription == 0:
      return torch.tensor(float('nan'))

    # Calculate V_half_prescription: volume receiving at least half the prescribed dose
    half_prescribed_dose = prescribed_dose / 2.0
    half_prescription_mask = dose >= half_prescribed_dose
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
  4. MAE formula: sum(|diff|) / count(ref > 5Gy in body)
  """
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.add_state("total_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("ref_5gy_count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor,
             ptv_doses: torch.Tensor, body_mask: torch.Tensor):
    """
    Update the metric with new predictions and targets.

    Args:
      preds: predicted doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
      target: reference doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
      ptv_doses: PTV doses, shape (batch_size, 3, H, W) or (batch_size, 3, H, W, D)
                 Index 0 = highest dose (used for D97 normalization)
      body_mask: Body structure mask, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred = preds[i, 0]  # Remove channel dimension
      ref_dose = target[i, 0]  # Remove channel dimension
      ptv_high_mask = (ptv_doses[i, 0] > 0).to(torch.bool)  # PTV_High is index 0
      body = body_mask[i, 0].to(torch.bool) if body_mask.dim() > 3 else body_mask[i].to(torch.bool)

      # Get prescribed dose for this patient from ptv_doses
      prescribed_dose = ptv_doses[i, 0].max() * GDPHMM_DOSE_MAX

      # Step 1: D97 normalization
      # Extract dose values within PTV_High
      ptv_dose_values = ref_dose[ptv_high_mask]

      if ptv_dose_values.numel() == 0:
        continue

      # Compute D97 (3rd percentile of dose within PTV_High)
      d97 = torch.quantile(ptv_dose_values, 1 - 0.97)

      # Compute normalization scale: prescribed_dose / D97
      norm_scale = prescribed_dose / (d97 + 1e-5)

      # Apply normalization to both reference and prediction
      ref_dose_normalized = ref_dose * norm_scale
      pred_normalized = pred * norm_scale

      # Step 2: Create masks
      # isodose_5gy_mask: regions where either reference or prediction > 5 Gy, within body
      isodose_5gy_mask = ((ref_dose_normalized > 5) | (pred_normalized > 5)) & body
      # isodose_ref_5gy_mask: regions where reference > 5 Gy, within body (for denominator)
      isodose_ref_5gy_mask = (ref_dose_normalized > 5) & body

      ref_count = isodose_ref_5gy_mask.sum()
      if ref_count == 0:
        continue

      # Step 3: Compute MAE following GDP-HMM formula
      # error = sum(|diff|) / count(ref > 5Gy in body)
      diff = torch.abs(ref_dose_normalized - pred_normalized)
      error_sum = diff[isodose_5gy_mask].sum()

      self.total_error += error_sum
      self.ref_5gy_count += ref_count

  def compute(self):
    """Compute the MAE following GDP-HMM formula."""
    if self.ref_5gy_count == 0:
      return torch.tensor(0.0)
    return self.total_error / self.ref_5gy_count


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
    # Create a metric for MSE on PTV region (union of multiple indices)
    ptv_mse = RegionMaskedMetric(MeanSquaredError(), region_name='PTV', region_index=[1, 2, 3])

    # Create a metric for PSNR on Brainstem region (single index)
    brainstem_psnr = RegionMaskedMetric(
      PeakSignalNoiseRatio(data_range=1.0),
      region_name='BrainStem_03',
      region_index=13
    )
  """
  def __init__(self, base_metric: Metric, region_name: str, region_index, *args, **kwargs):
    """
    Args:
      base_metric: The underlying metric to compute (e.g., MeanSquaredError, PSNR, etc.)
      region_name: Name of the region (e.g., 'PTV', 'BrainStem_03')
      region_index: Index or list of indices of the region in the structure tensor.
                    If a list is provided, a union mask is created from all indices.
    """
    super().__init__(*args, **kwargs)
    self.base_metric = base_metric
    self.region_name = region_name
    # Normalize to list for consistent handling
    self.region_indices = [region_index] if isinstance(region_index, int) else list(region_index)
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
      # Extract the region mask for this sample (union of all indices if multiple)
      region_mask = structure[i, self.region_indices[0]].to(torch.bool)
      for idx in self.region_indices[1:]:
        region_mask = region_mask | structure[i, idx].to(torch.bool)

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

  This metric computes MAE between predicted doses and the prescription level
  in the exclusive PTV region (not cumulative).

  Args:
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
  """
  use_cumulative_mask = False  # Use exclusive masks for prescription comparison

  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute MAE from prescription dose for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the exclusive PTV region
      prescribed_dose: Prescribed dose value in Gy

    Returns:
      MAE from prescription dose
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get dose values in this region
    doses = dose[region_mask]

    # Calculate mean absolute error from prescribed dose
    return torch.abs(doses - prescribed_dose).mean()


class PTVPrescriptionMeanDose(DoseLevelMetricBase):
  """
  Mean predicted dose in the exclusive region for a specific PTV level.

  This metric calculates the mean predicted dose in the exclusive region of a PTV level,
  allowing you to see how close predictions are to the prescription level.

  Args:
    ptv_level: The PTV level ('high', 'mid', or 'low'), or None for all levels
    compute_all_levels: If True, compute for all levels (high, mid, low)
  """
  use_cumulative_mask = False  # Use exclusive masks for prescription comparison

  def _compute_metric_for_sample(self, dose: torch.Tensor,
                                  region_mask: torch.Tensor, prescribed_dose: float) -> torch.Tensor:
    """
    Compute mean dose for a single sample at a specific dose level.

    Args:
      dose: Dose distribution for single sample
      region_mask: Boolean mask for the exclusive PTV region
      prescribed_dose: Prescribed dose value in Gy (unused for mean dose calculation)

    Returns:
      Mean dose in the region
    """
    if region_mask.sum() == 0:
      return torch.tensor(float('nan'))

    # Get dose values in this region
    doses = dose[region_mask]

    # Calculate mean dose
    return doses.mean()


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


class ConstraintSatisfactionRate(Metric):
  """
  Constraint Satisfaction Rate (CSR) - ratio of satisfied constraints (between 0 and 1).

  Computes CSR separately for predicted and target dose distributions, allowing
  comparison of how well each satisfies clinical constraints.

  Args:
    organ_names: List of organ names from GDPHMM_OAR
    voxel_spacing: Voxel spacing in mm (x, y, z). Default: (2.5, 2.5, 2.0)
    fractionation: Fractionation type. Default: CONVENTIONAL
    grade_filter: Optional constraint grade filter (e.g., MANDATORY). If None, uses all constraints.
  """
  def __init__(
    self,
    organ_names: list,
    voxel_spacing: tuple = (2.5, 2.5, 2.0),
    fractionation: FractionationType = FractionationType.CONVENTIONAL,
    grade_filter: Optional[ConstraintGrade] = None,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)

    self.organ_names = organ_names
    self.voxel_spacing = voxel_spacing
    self.voxel_volume_cc = np.prod(voxel_spacing) / 1000.0  # Convert mm³ to cm³
    self.fractionation = fractionation
    self.grade_filter = grade_filter

    # Initialize constraint library and evaluator
    library = ConstraintLibrary()
    if grade_filter:
      # Filter constraints by grade
      filtered_constraints = [c for c in library.constraints if c.grade == grade_filter]
      self.evaluator = ConstraintEvaluator(filtered_constraints)
    else:
      self.evaluator = ConstraintEvaluator(library.constraints)

    self.add_state("total_csr_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("total_csr_target", default=torch.tensor(0.0), dist_reduce_fx="sum")
    self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, organ_masks: Dict[str, torch.Tensor]):
    """
    Update metric with new predictions and targets.

    Args:
      preds: Predicted dose, shape (batch_size, 1, H, W, D) or (batch_size, 1, H, W)
      target: Target dose, shape (batch_size, 1, H, W, D) or (batch_size, 1, H, W)
      organ_masks: Dict mapping organ names to binary masks, each shape (batch_size, H, W, D) or (batch_size, H, W)
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      # Extract single sample
      pred_dose = preds[i, 0].cpu().numpy()
      target_dose = target[i, 0].cpu().numpy()

      # Prepare organ masks dict for this sample
      sample_organ_masks = {}
      for organ_name in self.organ_names:
        if organ_name in organ_masks:
          mask = organ_masks[organ_name][i].cpu().numpy()
          # Map to constraint organ name
          constraint_organ = GDPHMM_TO_CONSTRAINT_ORGAN.get(organ_name)
          if constraint_organ:
            sample_organ_masks[constraint_organ] = mask

      if len(sample_organ_masks) == 0:
        continue

      # Evaluate constraints for predicted dose
      pred_results = self.evaluator.evaluate_all_constraints(
        pred_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )

      # Evaluate constraints for target dose
      target_results = self.evaluator.evaluate_all_constraints(
        target_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )

      # Count satisfied constraints
      pred_satisfied = sum(
        sum(1 for r in organ_results if r['satisfied'])
        for organ_results in pred_results.values()
      )
      pred_total = sum(len(organ_results) for organ_results in pred_results.values())

      target_satisfied = sum(
        sum(1 for r in organ_results if r['satisfied'])
        for organ_results in target_results.values()
      )
      target_total = sum(len(organ_results) for organ_results in target_results.values())

      # Compute CSR (between 0 and 1)
      if pred_total > 0:
        csr_pred = pred_satisfied / pred_total
        self.total_csr_pred += csr_pred

      if target_total > 0:
        csr_target = target_satisfied / target_total
        self.total_csr_target += csr_target

      if pred_total > 0 and target_total > 0:
        self.count += 1

  def compute(self):
    """Compute average CSR for predictions and targets."""
    if self.count == 0:
      return {
        'csr_pred': torch.tensor(float('nan')),
        'csr_target': torch.tensor(float('nan')),
        'csr_diff_abs': torch.tensor(float('nan'))
      }

    csr_pred = self.total_csr_pred / self.count
    csr_target = self.total_csr_target / self.count

    return {
      'csr_pred': csr_pred,
      'csr_target': csr_target,
      'csr_diff_abs': torch.abs(csr_pred - csr_target)
    }


class ConstraintAgreementScore(Metric):
  """
  Constraint Agreement Score - ratio of constraints where pred and target have same satisfaction status.

  This metric measures how often the prediction and target agree on whether a constraint
  is satisfied or violated. A score of 1.0 means perfect agreement, 0.0 means no agreement.

  Can be computed overall or filtered by constraint grade (mandatory, optimal, etc.).

  Args:
    organ_names: List of organ names from GDPHMM_OAR
    voxel_spacing: Voxel spacing in mm (x, y, z). Default: (2.5, 2.5, 2.0)
    fractionation: Fractionation type. Default: CONVENTIONAL
    grade_filter: Optional constraint grade filter. If None, computes for all grades separately.
  """
  def __init__(
    self,
    organ_names: list,
    voxel_spacing: tuple = (2.5, 2.5, 2.0),
    fractionation: FractionationType = FractionationType.CONVENTIONAL,
    grade_filter: Optional[ConstraintGrade] = None,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)

    self.organ_names = organ_names
    self.voxel_spacing = voxel_spacing
    self.voxel_volume_cc = np.prod(voxel_spacing) / 1000.0
    self.fractionation = fractionation
    self.grade_filter = grade_filter

    # Initialize constraint library and evaluator
    library = ConstraintLibrary()
    self.evaluator = ConstraintEvaluator(library.constraints)

    # Track agreement for all grades if no filter
    if grade_filter is None:
      self.add_state("total_agreement_mandatory", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_mandatory", default=torch.tensor(0), dist_reduce_fx="sum")
      self.add_state("total_agreement_recommended", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_recommended", default=torch.tensor(0), dist_reduce_fx="sum")
      self.add_state("total_agreement_optimal", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_optimal", default=torch.tensor(0), dist_reduce_fx="sum")
      self.add_state("total_agreement_all", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count_all", default=torch.tensor(0), dist_reduce_fx="sum")
    else:
      self.add_state("total_agreement", default=torch.tensor(0.0), dist_reduce_fx="sum")
      self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, target: torch.Tensor, organ_masks: Dict[str, torch.Tensor]):
    """
    Update metric with new predictions and targets.

    Args:
      preds: Predicted dose
      target: Target dose
      organ_masks: Dict mapping organ names to binary masks
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred_dose = preds[i, 0].cpu().numpy()
      target_dose = target[i, 0].cpu().numpy()

      # Prepare organ masks
      sample_organ_masks = {}
      for organ_name in self.organ_names:
        if organ_name in organ_masks:
          mask = organ_masks[organ_name][i].cpu().numpy()
          constraint_organ = GDPHMM_TO_CONSTRAINT_ORGAN.get(organ_name)
          if constraint_organ:
            sample_organ_masks[constraint_organ] = mask

      if len(sample_organ_masks) == 0:
        continue

      # Evaluate constraints
      pred_results = self.evaluator.evaluate_all_constraints(
        pred_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )
      target_results = self.evaluator.evaluate_all_constraints(
        target_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )

      # Compare satisfaction status for each constraint
      if self.grade_filter is None:
        # Track by grade
        agreements = {
          ConstraintGrade.MANDATORY: [],
          ConstraintGrade.RECOMMENDED: [],
          ConstraintGrade.OPTIMAL: [],
          'all': []
        }

        for organ_name in pred_results.keys():
          if organ_name not in target_results:
            continue

          pred_organ = pred_results[organ_name]
          target_organ = target_results[organ_name]

          # Match constraints by type and threshold
          for pred_result in pred_organ:
            constraint = pred_result['constraint']
            # Find matching constraint in target results
            for target_result in target_organ:
              target_constraint = target_result['constraint']
              if (constraint.constraint_type == target_constraint.constraint_type and
                  constraint.threshold_value == target_constraint.threshold_value and
                  constraint.limit_value == target_constraint.limit_value):
                # Check if satisfaction status agrees
                agree = pred_result['satisfied'] == target_result['satisfied']
                agreements['all'].append(agree)
                if constraint.grade:
                  agreements[constraint.grade].append(agree)
                break

        # Update states
        if len(agreements['all']) > 0:
          agreement_rate_all = sum(agreements['all']) / len(agreements['all'])
          self.total_agreement_all += agreement_rate_all
          self.count_all += 1

        if len(agreements[ConstraintGrade.MANDATORY]) > 0:
          agreement_rate = sum(agreements[ConstraintGrade.MANDATORY]) / len(agreements[ConstraintGrade.MANDATORY])
          self.total_agreement_mandatory += agreement_rate
          self.count_mandatory += 1

        if len(agreements[ConstraintGrade.RECOMMENDED]) > 0:
          agreement_rate = sum(agreements[ConstraintGrade.RECOMMENDED]) / len(agreements[ConstraintGrade.RECOMMENDED])
          self.total_agreement_recommended += agreement_rate
          self.count_recommended += 1

        if len(agreements[ConstraintGrade.OPTIMAL]) > 0:
          agreement_rate = sum(agreements[ConstraintGrade.OPTIMAL]) / len(agreements[ConstraintGrade.OPTIMAL])
          self.total_agreement_optimal += agreement_rate
          self.count_optimal += 1

      else:
        # Single grade filter
        agreements = []
        for organ_name in pred_results.keys():
          if organ_name not in target_results:
            continue

          pred_organ = pred_results[organ_name]
          target_organ = target_results[organ_name]

          for pred_result in pred_organ:
            constraint = pred_result['constraint']
            if constraint.grade == self.grade_filter:
              for target_result in target_organ:
                target_constraint = target_result['constraint']
                if (constraint.constraint_type == target_constraint.constraint_type and
                    constraint.threshold_value == target_constraint.threshold_value and
                    constraint.limit_value == target_constraint.limit_value):
                  agree = pred_result['satisfied'] == target_result['satisfied']
                  agreements.append(agree)
                  break

        if len(agreements) > 0:
          agreement_rate = sum(agreements) / len(agreements)
          self.total_agreement += agreement_rate
          self.count += 1

  def compute(self):
    """Compute agreement scores."""
    if self.grade_filter is None:
      # Return scores for all grades
      results = {}

      if self.count_all > 0:
        results['agreement_all'] = self.total_agreement_all / self.count_all
      else:
        results['agreement_all'] = torch.tensor(float('nan'))

      if self.count_mandatory > 0:
        results['agreement_mandatory'] = self.total_agreement_mandatory / self.count_mandatory
      else:
        results['agreement_mandatory'] = torch.tensor(float('nan'))

      if self.count_recommended > 0:
        results['agreement_recommended'] = self.total_agreement_recommended / self.count_recommended
      else:
        results['agreement_recommended'] = torch.tensor(float('nan'))

      if self.count_optimal > 0:
        results['agreement_optimal'] = self.total_agreement_optimal / self.count_optimal
      else:
        results['agreement_optimal'] = torch.tensor(float('nan'))

      return results
    else:
      # Return single score for filtered grade
      if self.count > 0:
        return self.total_agreement / self.count
      else:
        return torch.tensor(float('nan'))


class PerOrganConstraintCompliance(Metric):
  """
  Per-organ constraint compliance - satisfaction rate for a single organ with breakdown by grade.

  Computes CSR for each grade level and compares pred vs target.
  This metric is designed to be instantiated once per organ/region.

  Args:
    organ_name: Single organ name from GDPHMM_OAR
    voxel_spacing: Voxel spacing in mm (x, y, z). Default: (2.5, 2.5, 2.0)
    fractionation: Fractionation type. Default: CONVENTIONAL
  """
  def __init__(
    self,
    organ_name: str,
    voxel_spacing: tuple = (2.5, 2.5, 2.0),
    fractionation: FractionationType = FractionationType.CONVENTIONAL,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)

    self.organ_name = organ_name
    self.voxel_spacing = voxel_spacing
    self.voxel_volume_cc = np.prod(voxel_spacing) / 1000.0
    self.fractionation = fractionation

    # Initialize constraint library and evaluator
    library = ConstraintLibrary()
    self.evaluator = ConstraintEvaluator(library.constraints)

    # Track per-grade statistics
    # Structure: {grade: {pred_satisfied: [], pred_total: [], target_satisfied: [], target_total: []}}
    self.grade_results = {}

  def update(self, preds: torch.Tensor, target: torch.Tensor, organ_masks: Dict[str, torch.Tensor]):
    """
    Update metric with new predictions and targets.

    Args:
      preds: Predicted dose
      target: Target dose
      organ_masks: Dict mapping organ names to binary masks
    """
    batch_size = preds.shape[0]

    for i in range(batch_size):
      pred_dose = preds[i, 0].cpu().numpy()
      target_dose = target[i, 0].cpu().numpy()

      # Check if this organ is in the organ_masks
      if self.organ_name not in organ_masks:
        continue

      # Get the mask for this organ
      mask = organ_masks[self.organ_name][i].cpu().numpy()
      constraint_organ = GDPHMM_TO_CONSTRAINT_ORGAN.get(self.organ_name)

      if not constraint_organ:
        continue

      # Prepare organ masks dict with single organ
      sample_organ_masks = {constraint_organ: mask}

      # Evaluate constraints
      pred_results = self.evaluator.evaluate_all_constraints(
        pred_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )
      target_results = self.evaluator.evaluate_all_constraints(
        target_dose, sample_organ_masks, self.voxel_volume_cc, self.fractionation
      )

      # Get results for this organ
      if constraint_organ not in pred_results or constraint_organ not in target_results:
        continue

      pred_organ_results = pred_results[constraint_organ]
      target_organ_results = target_results[constraint_organ]

      # Group results by grade
      grade_stats = {}

      # Count constraints by grade for predictions
      for pred_result in pred_organ_results:
        constraint = pred_result['constraint']
        grade = constraint.grade if constraint.grade else 'ungraded'

        if grade not in grade_stats:
          grade_stats[grade] = {
            'pred_satisfied': 0,
            'pred_total': 0,
            'target_satisfied': 0,
            'target_total': 0
          }

        grade_stats[grade]['pred_satisfied'] += 1 if pred_result['satisfied'] else 0
        grade_stats[grade]['pred_total'] += 1

      # Count constraints by grade for targets
      for target_result in target_organ_results:
        constraint = target_result['constraint']
        grade = constraint.grade if constraint.grade else 'ungraded'

        if grade not in grade_stats:
          grade_stats[grade] = {
            'pred_satisfied': 0,
            'pred_total': 0,
            'target_satisfied': 0,
            'target_total': 0
          }

        grade_stats[grade]['target_satisfied'] += 1 if target_result['satisfied'] else 0
        grade_stats[grade]['target_total'] += 1

      # Store results for each grade
      for grade, stats in grade_stats.items():
        if grade not in self.grade_results:
          self.grade_results[grade] = {
            'pred_satisfied': [],
            'pred_total': [],
            'target_satisfied': [],
            'target_total': []
          }

        self.grade_results[grade]['pred_satisfied'].append(stats['pred_satisfied'])
        self.grade_results[grade]['pred_total'].append(stats['pred_total'])
        self.grade_results[grade]['target_satisfied'].append(stats['target_satisfied'])
        self.grade_results[grade]['target_total'].append(stats['target_total'])

  def compute(self):
    """Compute per-grade CSR and differences for this organ."""
    results = {}

    # Compute CSR for each grade
    for grade, grade_data in self.grade_results.items():
      if len(grade_data['pred_total']) == 0:
        continue

      # Compute average CSR across all samples for this grade
      pred_csr_values = [
        s / t if t > 0 else 0.0
        for s, t in zip(grade_data['pred_satisfied'], grade_data['pred_total'])
      ]
      target_csr_values = [
        s / t if t > 0 else 0.0
        for s, t in zip(grade_data['target_satisfied'], grade_data['target_total'])
      ]

      pred_csr_mean = torch.tensor(np.mean(pred_csr_values))
      target_csr_mean = torch.tensor(np.mean(target_csr_values))

      # Convert grade enum to string for key names
      grade_key = grade.value if isinstance(grade, ConstraintGrade) else str(grade)

      # Flatten the results with grade suffix
      results[f'csr_pred_{grade_key}'] = pred_csr_mean
      results[f'csr_target_{grade_key}'] = target_csr_mean
      results[f'csr_abs_diff_{grade_key}'] = torch.abs(pred_csr_mean - target_csr_mean)

    return results
