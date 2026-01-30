import torch
import torch.nn as nn

from data import (
  GDPHMM_PTV_SDM_MIN, GDPHMM_PTV_SDM_MAX,
  GDPHMM_OAR_SDM_MIN, GDPHMM_OAR_SDM_MAX,
  GDPHMM_DOSE_MAX
)


def denormalize_ptv_sdm(ptv_sdm_normalized):
  """
  Convert normalized PTV SDM [-1, 1] back to mm [-50, 50].

  Args:
    ptv_sdm_normalized: Normalized PTV SDM tensor

  Returns:
    PTV SDM in mm (positive inside PTV, negative outside)
  """
  # normalized = (raw - min) / (max - min) * 2 - 1
  # raw = (normalized + 1) / 2 * (max - min) + min
  range_mm = GDPHMM_PTV_SDM_MAX - GDPHMM_PTV_SDM_MIN  # 100
  return (ptv_sdm_normalized + 1) / 2 * range_mm + GDPHMM_PTV_SDM_MIN


def denormalize_oar_sdm(oar_sdm_normalized):
  """
  Convert normalized OAR SDM [0, 1] back to mm [0, 40].

  Args:
    oar_sdm_normalized: Normalized OAR SDM tensor

  Returns:
    OAR SDM in mm (positive inside OAR, 0 outside)
  """
  # normalized = (raw - min) / (max - min)
  range_mm = GDPHMM_OAR_SDM_MAX - GDPHMM_OAR_SDM_MIN  # 40
  return oar_sdm_normalized * range_mm + GDPHMM_OAR_SDM_MIN



class AnalyticalDoseModel(nn.Module):
  """
  Analytical dose estimation model without learnable parameters.

  Uses a physics-inspired approach:
  1. Initialize dose to prescribed_dose inside PTV
  2. Outside PTV, dose falls off exponentially with distance
  3. Apply OAR sparing: reduce dose inside OARs based on depth

  Can be used as a standard model via build_model() or as a warm-start model
  via WarmStartModel wrapper (with output_uncertainty=True).

  Args:
    structure_selector: StructureSelector defining which groups are available
    falloff_constant: Exponential falloff constant in mm (default: 25.0)
    sparing_radius: OAR sparing radius in mm (default: 15.0)
    output_normalized: If True, output dose normalized to [0, 1] (default: True)
    output_uncertainty: If True, output [B, 2, H, W] with uncertainty channel (default: False)
    n_samples: Number of Monte Carlo samples for aleatoric uncertainty (default: 16)
    falloff_range: (min, max) uniform sampling range for falloff_constant in MC (default: (20.0, 35.0))
    sparing_range: (min, max) uniform sampling range for sparing_radius in MC (default: (10.0, 20.0))
  """

  def __init__(
    self,
    structure_selector,
    falloff_constant=25.0,
    sparing_radius=15.0,
    output_normalized=True,
    output_uncertainty=False,
    n_samples=16,
    falloff_range=(20.0, 35.0),
    sparing_range=(10.0, 20.0),
  ):
    super().__init__()
    self.selector = structure_selector
    self.falloff_constant = falloff_constant
    self.sparing_radius = sparing_radius
    self.output_normalized = output_normalized
    self.output_uncertainty = output_uncertainty
    self.n_samples = n_samples
    self.falloff_range = falloff_range
    self.sparing_range = sparing_range

    # Validate required groups
    required_groups = ['ptv_doses', 'ptv_sdm', 'oar_v3_sdm']
    for group in required_groups:
      if group not in structure_selector.groups:
        raise ValueError(f"StructureSelector must include '{group}' group")

  def forward(self, condition):
    """
    Compute analytical dose estimate from conditioning.

    Args:
      condition: Conditioning tensor [B, C, H, W] containing ptv_doses, ptv_sdm,
                 and oar_v3_sdm

    Returns:
      If output_uncertainty is False:
        Dose tensor [B, 1, H, W]
      If output_uncertainty is True:
        Tensor [B, 2, H, W] where channel 0 is dose and channel 1 is
        softplus_inverse(uncertainty), compatible with WarmStartModel wrapper
    """
    # Extract PTV doses (normalized by GDPHMM_DOSE_MAX) and SDMs
    ptv_doses = self.selector.get_ptv_doses(condition)  # [B, 3, H, W], values in [0, 1]
    ptv_sdm_norm = self.selector.get_component(condition, 'ptv_sdm')  # [B, 3, H, W], normalized [-1, 1]

    # Convert PTV SDM to mm
    ptv_sdm_mm = denormalize_ptv_sdm(ptv_sdm_norm)  # [B, 3, H, W], mm

    # Convert prescribed doses to Gy
    prescribed_doses_gy = ptv_doses * GDPHMM_DOSE_MAX  # [B, 3, H, W]

    # Compute main dose estimate with default parameters
    dose = self._compute_dose_core(
      condition, ptv_sdm_mm, prescribed_doses_gy,
      self.falloff_constant, self.sparing_radius
    )

    if not self.output_uncertainty:
      if self.output_normalized:
        dose = dose / GDPHMM_DOSE_MAX
      return dose

    # Compute uncertainty
    uncertainty = self._compute_uncertainty(
      condition, dose, ptv_sdm_mm, prescribed_doses_gy
    )

    # Normalize output
    if self.output_normalized:
      dose = dose / GDPHMM_DOSE_MAX
      uncertainty = uncertainty / GDPHMM_DOSE_MAX

    # Encode uncertainty as log_sigma for WarmStartModel compatibility:
    # WarmStartModel computes sigma = softplus(log_sigma) + min_sigma
    # softplus_inverse(x) = x + log(1 - exp(-x))  [numerically stable for x > 0]
    log_sigma = uncertainty + torch.log1p(-torch.exp(-uncertainty))
    log_sigma = torch.clamp(log_sigma, min=-20.0)

    return torch.cat([dose, log_sigma], dim=1)

  def _compute_dose_core(self, condition, ptv_sdm_mm, prescribed_doses_gy, falloff_constant, sparing_radius):
    """
    Compute dose distribution with given parameters.

    Args:
      condition: Conditioning tensor [B, C, H, W]
      ptv_sdm_mm: PTV signed distance maps in mm [B, 3, H, W]
      prescribed_doses_gy: Prescribed doses in Gy [B, 3, H, W]
      falloff_constant: Exponential falloff constant in mm
      sparing_radius: OAR sparing radius in mm

    Returns:
      dose: Dose estimate [B, 1, H, W] in Gy
    """
    B, _, H, W = condition.shape
    dose = torch.zeros(B, 1, H, W, device=condition.device, dtype=condition.dtype)

    for i in range(3):
      ptv_dose = prescribed_doses_gy[:, i:i+1]  # [B, 1, H, W]
      sdm_mm = ptv_sdm_mm[:, i:i+1]  # [B, 1, H, W]

      if ptv_dose.max() == 0:
        continue

      inside_mask = sdm_mm > 0
      dose_contribution = torch.where(inside_mask, ptv_dose, torch.zeros_like(ptv_dose))

      outside_mask = sdm_mm < 0
      distance_outside = -sdm_mm
      falloff = torch.exp(-distance_outside / falloff_constant)
      dose_contribution = torch.where(
        outside_mask,
        ptv_dose * falloff,
        dose_contribution
      )

      dose = torch.maximum(dose, dose_contribution)

    dose = self._apply_oar_sparing(condition, dose, sparing_radius)
    return dose

  def _compute_uncertainty(self, condition, dose, ptv_sdm_mm, prescribed_doses_gy):
    """
    Compute combined aleatoric and epistemic uncertainty.

    Aleatoric: Monte Carlo sampling with varied falloff_constant and sparing_radius.
    Epistemic: Spatially varying, increases with distance from PTV boundary.

    Args:
      condition: Conditioning tensor [B, C, H, W]
      dose: Main dose estimate [B, 1, H, W] in Gy
      ptv_sdm_mm: PTV signed distance maps in mm [B, 3, H, W]
      prescribed_doses_gy: Prescribed doses in Gy [B, 3, H, W]

    Returns:
      uncertainty: Combined uncertainty [B, 1, H, W] in Gy
    """
    B, _, H, W = condition.shape

    # 1. Aleatoric uncertainty via Monte Carlo parameter sampling
    mc_samples = []
    for _ in range(self.n_samples):
      fc = torch.empty(1).uniform_(*self.falloff_range).item()
      sr = torch.empty(1).uniform_(*self.sparing_range).item()
      sample = self._compute_dose_core(
        condition, ptv_sdm_mm, prescribed_doses_gy, fc, sr
      )
      mc_samples.append(sample)

    mc_stack = torch.stack(mc_samples, dim=0)  # [n_samples, B, 1, H, W]
    aleatoric_std = mc_stack.std(dim=0)  # [B, 1, H, W]

    # 2. Epistemic uncertainty (spatially varying - higher far from PTV)
    # Use minimum absolute SDM across active PTV levels as distance from nearest PTV
    min_abs_sdm = torch.full(
      (B, 1, H, W), float('inf'),
      device=condition.device, dtype=condition.dtype
    )
    for i in range(3):
      if prescribed_doses_gy[:, i:i+1].max() > 0:
        abs_sdm = ptv_sdm_mm[:, i:i+1].abs()
        min_abs_sdm = torch.minimum(min_abs_sdm, abs_sdm)

    # Replace inf for voxels with no active PTV
    min_abs_sdm = torch.where(
      min_abs_sdm.isinf(), torch.zeros_like(min_abs_sdm), min_abs_sdm
    )

    base_factor = 0.05
    length_scale = 50.0
    epistemic = base_factor * dose * (1 + min_abs_sdm / length_scale)
    epistemic = torch.clamp(epistemic, min=torch.zeros_like(epistemic), max=0.3 * dose)

    # 3. Combine uncertainties
    uncertainty = torch.sqrt(aleatoric_std ** 2 + epistemic ** 2)

    return uncertainty

  def _apply_oar_sparing(self, condition, dose, sparing_radius=None):
    """
    Reduce dose inside OARs based on depth.

    Args:
      condition: Conditioning tensor [B, C, H, W]
      dose: Current dose estimate [B, 1, H, W] in Gy
      sparing_radius: OAR sparing radius in mm (uses self.sparing_radius if None)

    Returns:
      dose: Dose with OAR sparing applied [B, 1, H, W]
    """
    if sparing_radius is None:
      sparing_radius = self.sparing_radius

    # Get OAR SDM
    oar_sdm_norm = self.selector.get_component(condition, 'oar_v3_sdm')  # [B, N_OAR, H, W]

    # Convert to mm
    oar_sdm_mm = denormalize_oar_sdm(oar_sdm_norm)  # [B, N_OAR, H, W]

    # For each OAR, compute protection factor
    # Inside OAR (sdm > 0): reduce dose based on depth
    # Protection factor = clip(sdm_mm / sparing_radius, 0, 1)
    # Dose multiplier = 1 - protection_factor (deeper inside = more protection = less dose)

    # Aggregate across all OARs: use maximum protection (minimum multiplier)
    # protection = max over OARs of clip(sdm_mm / sparing_radius, 0, 1)
    protection_per_oar = torch.clamp(oar_sdm_mm / sparing_radius, 0, 1)  # [B, N_OAR, H, W]
    max_protection = protection_per_oar.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]

    # Apply protection: multiply dose by (1 - protection)
    dose_multiplier = 1.0 - max_protection
    dose = dose * dose_multiplier

    return dose


