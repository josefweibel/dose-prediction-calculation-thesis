"""
Utilities for handling conditional inputs (masks, SDMs, etc.) for dose prediction models.
"""

import torch
from data import GDPHMM_OAR


class ConditioningSelector:
  """
  Handles selection and concatenation of conditioning inputs based on configuration.

  Supported condition types:
  - 'ct': CT image
  - 'ptv_mask': PTV binary mask
  - 'oar_mask': OAR binary masks (all channels)
  - 'ptv_sdm': PTV signed distance map
  - 'oar_sdm': OAR signed distance maps (all channels)
  """

  def __init__(self, condition_types, n_oar_channels=None):
    """
    Args:
      condition_types: List of condition types to use (e.g., ['ct', 'ptv_mask', 'oar_mask'])
      n_oar_channels: Number of OAR channels (default: len(GDPHMM_OAR))
    """
    self.condition_types = condition_types
    self.n_oar_channels = n_oar_channels if n_oar_channels is not None else len(GDPHMM_OAR)

    # Validate condition types
    valid_types = {'ct', 'ptv_mask', 'oar_mask', 'ptv_sdm', 'oar_sdm'}
    for ctype in condition_types:
      if ctype not in valid_types:
        raise ValueError(f"Invalid condition type '{ctype}'. Must be one of {valid_types}")

    # Calculate channel indices for each condition type
    self.channel_map = self._build_channel_map()
    self.n_input_channels = self._calculate_n_input_channels()

  def _build_channel_map(self):
    """Build a mapping of condition types to their channel indices in the input tensor."""
    channel_map = {}

    # Standard structure without SDM: [CT (1), PTV (1), OARs (N)]
    # With SDM: [CT (1), PTV (1), OARs (N), PTV_SDM (1), OAR_SDMs (N)]

    ct_idx = 0
    ptv_mask_idx = 1
    oar_mask_start = 2
    oar_mask_end = 2 + self.n_oar_channels
    ptv_sdm_idx = oar_mask_end
    oar_sdm_start = oar_mask_end + 1
    oar_sdm_end = oar_mask_end + 1 + self.n_oar_channels

    channel_map['ct'] = slice(ct_idx, ct_idx + 1)
    channel_map['ptv_mask'] = slice(ptv_mask_idx, ptv_mask_idx + 1)
    channel_map['oar_mask'] = slice(oar_mask_start, oar_mask_end)
    channel_map['ptv_sdm'] = slice(ptv_sdm_idx, ptv_sdm_idx + 1)
    channel_map['oar_sdm'] = slice(oar_sdm_start, oar_sdm_end)

    return channel_map

  def _calculate_n_input_channels(self):
    """Calculate the total number of input channels based on selected conditions."""
    n_channels = 0
    for ctype in self.condition_types:
      if ctype == 'ct':
        n_channels += 1
      elif ctype == 'ptv_mask':
        n_channels += 1
      elif ctype == 'oar_mask':
        n_channels += self.n_oar_channels
      elif ctype == 'ptv_sdm':
        n_channels += 1
      elif ctype == 'oar_sdm':
        n_channels += self.n_oar_channels
    return n_channels

  def select_channels(self, structure_tensor):
    """
    Select and concatenate the specified conditioning channels from the input tensor.

    Args:
      structure_tensor: Input tensor of shape (B, C, H, W) where C includes all channels
                        Format: [CT, PTV_mask, OAR_masks, PTV_SDM, OAR_SDMs] (if SDM enabled)

    Returns:
      Selected tensor of shape (B, C', H, W) where C' is determined by condition_types
    """
    selected = []

    for ctype in self.condition_types:
      channel_slice = self.channel_map[ctype]
      selected.append(structure_tensor[:, channel_slice])

    return torch.cat(selected, dim=1)

  def get_n_input_channels(self):
    """Return the total number of input channels after selection."""
    return self.n_input_channels

  def needs_sdm(self):
    """Check if any SDM-based conditioning is requested."""
    return 'ptv_sdm' in self.condition_types or 'oar_sdm' in self.condition_types

  def __repr__(self):
    return (f"ConditioningSelector(condition_types={self.condition_types}, "
            f"n_input_channels={self.n_input_channels})")


def build_conditioning_selector(config):
  """
  Build a conditioning selector from config.

  Args:
    config: Model configuration dict with optional 'conditioning' section

  Returns:
    ConditioningSelector instance
  """
  # Default: use all mask-based conditioning (CT + PTV + OARs)
  default_conditions = ['ct', 'ptv_mask', 'oar_mask']

  condition_types = config.get('conditioning', {}).get('types', default_conditions)

  return ConditioningSelector(
    condition_types=condition_types,
    n_oar_channels=len(GDPHMM_OAR)
  )
