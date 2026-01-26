import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
import pandas as pd

GDPHMM_DATA_BASE_PATH = Path(__file__).parents[1] / 'data'

GDPHMM_IMG_MIN = -1024
GDPHMM_IMG_MAX = 3071

GDPHMM_DOSE_MIN = 0
GDPHMM_DOSE_MAX = 88

# SDM (Signed Distance Map) clipping and normalization ranges
GDPHMM_OAR_SDM_MIN = 0
GDPHMM_OAR_SDM_MAX = 40  # mm

GDPHMM_PTV_SDM_MIN = -50  # mm
GDPHMM_PTV_SDM_MAX = 50   # mm

# Beam plate normalization range
GDPHMM_BEAM_PLATE_MIN = 0
GDPHMM_BEAM_PLATE_MAX = 92

GDPHMM_OAR = [
  'Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R',
  'Chiasm', 'LacrimalGlands', 'BrachialPlexus', 'Brain', 'BrainStem_03', 'Esophagus',
  'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV',
  'OCavity-PTV', 'ParotidCon-PTV', 'ParotidIps-PTV', 'Parotids-PTV', 'PharConst-PTV',
  'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05',
  'Pituitary', 'OralCavity', 'Submandibular', # additional
]

# Version 2: Remove last three values (Pituitary, OralCavity, Submandibular)
GDPHMM_OAR_V2 = GDPHMM_OAR[:-3]

GDPHMM_OAR_V3 = [
  'Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R',
  'Chiasm', 'Brain', 'BrainStem_03', 'Esophagus', # dropped 'LacrimalGlands', 'BrachialPlexus',
  'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV',
  'OCavity-PTV', 'ParotidCon-PTV', 'ParotidIps-PTV', 'Parotids-PTV', 'PharConst-PTV',
  'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05',
]



class StructureSelector:
  """
  Helper class to manage selective loading and indexing of structure components.

  This class solves two problems:
  1. Allows loading only needed structure groups (saves memory/computation)
  2. Provides easy access to structure components without manual index tracking

  Usage:
    # Define what you need
    selector = StructureSelector(groups=['ct', 'oar', 'ptv_doses'])

    # In dataset __getitem__, load only selected groups
    structure = selector.build_structure(data)

    # Access components easily
    ct = selector.get_ct(structure)
    ptv_doses = selector.get_ptv_doses(structure)  # (3, H, W) - sorted by dose descending

  Available groups:
    - 'ct': CT scan (1 channel)
    - 'body': Body mask (1 channel)
    - 'ptv_doses': PTV masks sorted by dose descending (3 channels), values are dose/GDPHMM_DOSE_MAX
    - 'oar': OAR binary masks (33 channels)
    - 'oar_sdm': OAR signed distance maps (33 channels)
    - 'oar_v2': OAR binary masks without last 3 (30 channels)
    - 'oar_v2_sdm': OAR signed distance maps without last 3 (30 channels)
    - 'oar_v3': OAR binary masks without last 3 and 2 unused (28 channels)
    - 'oar_v3_sdm': OAR signed distance maps without last 3 and 2 unused (28 channels)
    - 'ptv_sdm': PTV signed distance maps (3 channels), one per PTV level
    - 'beam_plate': Beam configuration plate (1 channel)
    - 'angle_plate': Angle configuration plate (1 channel)
  """

  # Define all available structure groups and their channel counts
  AVAILABLE_GROUPS = {
    'ct': 1,
    'body': 1,
    'ptv_doses': 3,  # PTV masks sorted by dose descending
    'oar': len(GDPHMM_OAR),  # 33
    'oar_sdm': len(GDPHMM_OAR),  # 33
    'oar_v2': len(GDPHMM_OAR_V2),  # 30 (without last 3)
    'oar_v2_sdm': len(GDPHMM_OAR_V2),  # 30 (without last 3)
    'oar_v3': len(GDPHMM_OAR_V3),  # 28 (new version)
    'oar_v3_sdm': len(GDPHMM_OAR_V3),  # 28 (new version)
    'ptv_sdm': 3,  # PTV SDMs sorted by dose descending
    'beam_plate': 1,  # beam configuration plate
    'angle_plate': 1,  # angle configuration plate (3D around isocenter)
  }

  def __init__(self, groups=None):
    """
    Initialize structure selector.

    Args:
      groups: List of group names to include. If None, includes all groups.
              Example: ['ct', 'oar', 'ptv_sdm']
    """
    if groups is None:
      groups = list(self.AVAILABLE_GROUPS.keys())

    # Validate groups
    invalid = set(groups) - set(self.AVAILABLE_GROUPS.keys())
    if invalid:
      raise ValueError(f"Invalid groups: {invalid}. Available: {list(self.AVAILABLE_GROUPS.keys())}")

    self.groups = groups

    # Build index mapping: group_name -> (start_idx, end_idx)
    self._index_map = {}
    current_idx = 0
    for group in groups:
      n_channels = self.AVAILABLE_GROUPS[group]
      self._index_map[group] = (current_idx, current_idx + n_channels)
      current_idx += n_channels

    self.total_channels = current_idx

  def build_structure(self, data):
    """
    Build structure tensor from preprocessed data dict.

    Args:
      data: Dict with keys 'ct', 'body', 'ptv_doses', 'oar', 'oar_sdm', 'ptv_sdm',
            'beam_plate', 'angle_plate'
            Each value should be a torch.Tensor or numpy array

    Returns:
      structure: (C, H, W) tensor with only selected groups concatenated
    """
    components = []

    for group in self.groups:
      if group == 'ct':
        components.append(data['ct'])
      elif group == 'body':
        components.append(data['body'])
      elif group == 'ptv_doses':
        components.append(data['ptv_doses'])
      elif group == 'oar':
        components.append(data['oar'])
      elif group == 'oar_sdm':
        components.append(data['oar_sdm'])
      elif group == 'oar_v2':
        # Remove last 3 channels from oar
        components.append(data['oar'][:-3])
      elif group == 'oar_v2_sdm':
        # Remove last 3 channels from oar_sdm
        components.append(data['oar_sdm'][:-3])
      elif group == 'oar_v3':
        # Remove last 3 and indices 8 and 9 from oar
        components.append(data['oar'][[i for i in range(data['oar'].shape[0]) if i not in (8, 9, 30, 31, 32)]])
      elif group == 'oar_v3_sdm':
        # Remove last 3 and indices 8 and 9 from oar_sdm
        components.append(data['oar_sdm'][[i for i in range(data['oar_sdm'].shape[0]) if i not in (8, 9, 30, 31, 32)]])
      elif group == 'ptv_sdm':
        components.append(data['ptv_sdm'])
      elif group == 'beam_plate':
        components.append(data['beam_plate'])
      elif group == 'angle_plate':
        components.append(data['angle_plate'])

    # Concatenate all components
    structure = torch.cat(components, dim=0)
    return structure

  def get_component(self, structure, group_name):
    """
    Extract a specific component from the concatenated structure tensor.

    Args:
      structure: Concatenated structure tensor with one of the following shapes:
        - (C, H, W): Unbatched 2D slice
        - (B, C, H, W): Batched 2D slices
        - (B, C, H, W, D): Batched 3D volumes
      group_name: Name of group to extract (e.g., 'ct', 'oar')

    Returns:
      component: Tensor for the requested group with same batch/spatial dims as input
    """
    start, end = self.get_channel_indices(group_name)

    # Handle both batched and unbatched tensors
    if structure.ndim >= 4:  # (B, C, H, W) or (B, C, H, W, D)
      return structure[:, start:end]
    else:  # (C, H, W)
      return structure[start:end]

  def get_channel_indices(self, group_name):
    """
    Get the channel index range for a specific group.

    Args:
      group_name: Name of group

    Returns:
      (start_idx, end_idx): Tuple of start and end indices
    """
    if group_name not in self._index_map:
      raise ValueError(f"Group '{group_name}' not in selected groups: {self.groups}")
    return self._index_map[group_name]

  def get_oar_index(self, oar_name):
    """
    Get the channel index for a specific OAR within the OAR group.

    Args:
      oar_name: Name of the OAR

    Returns:
      Global channel index in the concatenated structure tensor
    """
    if 'oar_v3' in self._index_map:
      oar_idx = GDPHMM_OAR_V3.index(oar_name)
      oar_start, _ = self._index_map['oar_v3']
    elif 'oar_v2' in self._index_map:
      oar_idx = GDPHMM_OAR_V2.index(oar_name)
      oar_start, _ = self._index_map['oar_v2']
    elif 'oar' in self._index_map:
      oar_idx = GDPHMM_OAR.index(oar_name)
      oar_start, _ = self._index_map['oar']
    else:
      raise ValueError("Neither 'oar', 'oar_v2' nor 'oar_v3' group in selected groups")

    return oar_start + oar_idx

  def get_oar_sdm_index(self, oar_name):
    """
    Get the channel index for a specific OAR SDM within the OAR SDM group.

    Args:
      oar_name: Name of the OAR

    Returns:
      Global channel index in the concatenated structure tensor
    """
    if 'oar_v3_sdm' in self._index_map:
      oar_idx = GDPHMM_OAR_V3.index(oar_name)
      oar_sdm_start, _ = self._index_map['oar_v3_sdm']
    elif 'oar_v2_sdm' in self._index_map:
      oar_idx = GDPHMM_OAR_V2.index(oar_name)
      oar_sdm_start, _ = self._index_map['oar_v2_sdm']
    elif 'oar_sdm' in self._index_map:
      oar_idx = GDPHMM_OAR.index(oar_name)
      oar_sdm_start, _ = self._index_map['oar_sdm']
    else:
      raise ValueError("Neither 'oar_sdm', 'oar_v2_sdm' nor 'oar_v3_sdm' group in selected groups")

    return oar_sdm_start + oar_idx

  def select_channels(self, structure, groups):
    """
    Select specific groups from the structure tensor.

    This is useful when you have a full structure tensor but only need certain groups
    (e.g., for conditioning the model with a subset of available structures).

    Args:
      structure: (B, C, H, W) or (C, H, W) structure tensor
      groups: List of group names to select (e.g., ['ct', 'ptv'])

    Returns:
      Selected channels concatenated together

    Example:
      # Dataset loaded all: ['ct', 'ptv', 'oar']
      structure, _, _, _ = dataset[0]  # 35 channels

      # Select only ct and ptv for conditioning
      condition = dataset.selector.select_channels(structure, ['ct', 'ptv'])  # 2 channels
    """
    selected = []
    for group in groups:
      if group not in self._index_map:
        raise ValueError(f"Group '{group}' not in dataset's loaded groups: {self.groups}")
      selected.append(self.get_component(structure, group))

    return torch.cat(selected, dim=-3)  # Concatenate on channel dim (works for both 3D and 4D)

  def get_ptv_doses(self, structure):
    """
    Get PTV doses channels from structure tensor.

    Args:
      structure: Structure tensor - (C, H, W), (B, C, H, W), or (B, C, H, W, D)

    Returns:
      PTV doses channels: (3, H, W), (B, 3, H, W), or (B, 3, H, W, D)
    """
    return self.get_component(structure, 'ptv_doses')

  def get_ct(self, structure):
    """
    Get CT channel from structure tensor.

    Args:
      structure: Structure tensor - (C, H, W), (B, C, H, W), or (B, C, H, W, D)

    Returns:
      CT channel(s): (1, H, W), (B, 1, H, W), or (B, 1, H, W, D)
    """
    return self.get_component(structure, 'ct')

  def get_body(self, structure):
    """
    Get body mask channel from structure tensor.

    Args:
      structure: Structure tensor - (C, H, W), (B, C, H, W), or (B, C, H, W, D)

    Returns:
      Body mask channel(s): (1, H, W), (B, 1, H, W), or (B, 1, H, W, D)
    """
    return self.get_component(structure, 'body')

  def get_oar(self, structure, oar_name):
    """
    Get specific OAR channel from structure tensor.

    Args:
      structure: Structure tensor with one of the following shapes:
        - (C, H, W): Unbatched 2D slice
        - (B, C, H, W): Batched 2D slices
        - (B, C, H, W, D): Batched 3D volumes
      oar_name: Name of the OAR

    Returns:
      OAR channel: (1, H, W), (B, 1, H, W), or (B, 1, H, W, D)
    """
    if 'oar_v3' in self._index_map:
      oar_idx = GDPHMM_OAR_V3.index(oar_name)
      oar_start, _ = self._index_map['oar_v3']
    elif 'oar_v2' in self._index_map:
      oar_idx = GDPHMM_OAR_V2.index(oar_name)
      oar_start, _ = self._index_map['oar_v2']
    elif 'oar' in self._index_map:
      oar_idx = GDPHMM_OAR.index(oar_name)
      oar_start, _ = self._index_map['oar']
    else:
      raise ValueError("Neither 'oar', 'oar_v2' nor 'oar_v3' group in selected groups")

    global_idx = oar_start + oar_idx

    # Handle both batched and unbatched tensors
    if structure.ndim >= 4:  # (B, C, H, W) or (B, C, H, W, D)
      return structure[:, global_idx:global_idx+1]
    else:  # (C, H, W)
      return structure[global_idx:global_idx+1]

  def __repr__(self):
    return f"StructureSelector(groups={self.groups}, total_channels={self.total_channels})"


class GDPHMMPreprocessedSlicesDataset(Dataset):
  """
  Dataset for loading preprocessed GDPHMM slices with flexible structure selection.

  This dataset loads slices that have been preprocessed with all transformations
  already applied (clipping, normalization, SDM computation, type conversions, padding).
  No caching is needed since slices are stored individually and ready to use.

  Uses StructureSelector to:
    1. Load only needed structure groups (saves memory/computation)
    2. Provide easy access to structure components without manual indexing

  Compared to GDPHMMSlicesDataset:
    - No cache needed (slices already preprocessed)
    - Faster loading (no on-the-fly transformations or padding)
    - Simpler implementation
    - Lower memory usage
    - Flexible structure selection

  Args:
    dataset_name: Name of dataset (e.g., 'gdphmm-hn')
    split: 'train', 'valid', or 'test'
    structure_groups: List of structure groups to load. If None, loads all.
                      Available: ['ct', 'body', 'ptv_doses', 'oar', 'oar_sdm', 'ptv_sdm', 'beam_plate', 'angle_plate']
                      Example: ['ct', 'body', 'oar', 'ptv_sdm'] to load only CT, body mask, OARs, and PTV SDMs
    limit: Optional limit on number of slices
    skip: Number of slices to skip from start

  Note:
    Assumes slices were preprocessed with --pad-size argument.
    Preprocess your data with:
      python scripts/preprocess_gdphmm_slices.py --pad-size 256 256

  Example usage:
    # Load only CT, body, OARs, and PTV doses (saves memory compared to loading everything)
    dataset = GDPHMMPreprocessedSlicesDataset(
      'gdphmm-hn', 'train',
      structure_groups=['ct', 'body', 'oar', 'ptv_doses']
    )
    structure, dose, info = dataset[0]

    # Access specific components easily
    ct = dataset.selector.get_ct(structure)
    body = dataset.selector.get_body(structure)
    ptv_doses = dataset.selector.get_ptv_doses(structure)  # (3, H, W) - sorted by dose descending
  """
  def __init__(
    self,
    dataset_name,
    split,
    structure_groups=None,
    limit=None,
    skip=0,
  ):
    self.dataset_name = dataset_name
    self.split = split

    self.selector = StructureSelector(groups=structure_groups)

    # Load index of preprocessed slices
    data_dir = GDPHMM_DATA_BASE_PATH / f'{dataset_name}_slices' / split
    index_path = data_dir / '_index.csv'

    if not index_path.exists():
      raise FileNotFoundError(
        f"Preprocessed data not found at {data_dir}. "
        f"Run: python scripts/preprocess_gdphmm_slices.py --dataset {dataset_name} --pad-size 256 256"
      )

    self.index = pd.read_csv(index_path)
    self.data_dir = data_dir

    # Apply skip and limit
    if skip > 0:
      self.index = self.index.iloc[skip:]
    if limit is not None:
      indices = np.linspace(0, len(self.index) - 1, limit, dtype=int)
      self.index = self.index.iloc[indices]

    self.index = self.index.reset_index(drop=True)

    # Use selector's total channels instead of hardcoded value
    self.n_structures = self.selector.total_channels
    self.dose_range = (GDPHMM_DOSE_MIN, GDPHMM_DOSE_MAX)

  def __len__(self):
    return len(self.index)

  def __getitem__(self, idx):
    """
    Load a preprocessed slice (already padded) with only selected structure groups.

    Returns:
      structure: (C, H, W) tensor with C channels determined by structure_groups
                 Use dataset.selector.get_component(structure, 'ct') or 'body' to access components
      dose: (1, H, W) tensor, normalized dose
      info: dict with slice metadata
    """
    row = self.index.iloc[idx]
    slice_path = self.data_dir / row['file']

    # Load preprocessed slice (already transformed and padded)
    data_np = np.load(slice_path)

    # Convert all data to tensors with proper type conversions
    data = {
      'ct': torch.from_numpy(data_np['ct']),  # float32, normalized [0, 1]
      'body': torch.from_numpy(data_np['body']),  # uint8 {0, 1}
      'ptv_doses': torch.from_numpy(data_np['ptv_doses']).float() / GDPHMM_DOSE_MAX,  # uint8 → float32 [0, 1]
      'oar': torch.from_numpy(data_np['oar']),  # uint8 {0, 1}
      'oar_sdm': torch.from_numpy(data_np['oar_sdm']).float(),  # float16 → float32
      'ptv_sdm': torch.from_numpy(data_np['ptv_sdm']).float(),  # float16 → float32
      'beam_plate': torch.from_numpy(data_np['beam_plate']),  # float32, continuous values
      'angle_plate': torch.from_numpy(data_np['angle_plate']),  # uint8 {0, 1}, binary
    }

    # Use selector to build structure with only selected groups
    structure = self.selector.build_structure(data)

    # Dose is always loaded
    dose = torch.from_numpy(data_np['dose'])  # float32, normalized [0, 1]

    # Metadata
    info = {
      'file_path': str(slice_path),
      'file_name': slice_path.parent.with_suffix('.npz').name,
      'volume': row['volume'],
      'index': row['slice_idx']
    }

    return structure, dose, info

  def denormalize_target(self, x):
    """Denormalize dose from [0, 1] to [0, 88] Gy."""
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

  def finish(self):
    """Compatibility method (no cleanup needed for preprocessed dataset)."""
    pass


def create_gdphmm_dataset(config, split, structure_groups=None, force_complete=True):
  """
  Factory function to create GDPHMM dataset based on configuration.

  Args:
    config: Configuration dict
    split: 'train', 'valid', or 'test'
    structure_groups: List of structure groups to load. If None, loads all available groups.

  Config:
    data:
      dataset_name: Name of dataset
  """
  data_config = config['data']
  dataset_name = data_config['dataset_name']
  limit = None if force_complete else data_config.get(f'{split}_limit', None)
  skip = 0 if force_complete else data_config.get(f'{split}_skip', 0)

  return GDPHMMPreprocessedSlicesDataset(
    dataset_name=dataset_name,
    split=split,
    structure_groups=structure_groups,
    limit=limit,
    skip=skip,
  )
