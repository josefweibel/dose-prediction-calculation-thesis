import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from multiprocessing import shared_memory
import time
from scipy.ndimage import distance_transform_edt

GDPHMM_DATA_BASE_PATH = Path(__file__).parents[1] / 'data'

GDPHMM_IMG_MIN = -1024
GDPHMM_IMG_MAX = 3071

GDPHMM_DOSE_MIN = 0
GDPHMM_DOSE_MAX = 88

GDPHMM_OAR = [
  'Cochlea_L', 'Cochlea_R', 'Eyes', 'Lens_L', 'Lens_R', 'OpticNerve_L', 'OpticNerve_R',
  'Chiasm', 'LacrimalGlands', 'BrachialPlexus', 'Brain', 'BrainStem_03', 'Esophagus',
  'Lips', 'Lungs', 'Trachea', 'Posterior_Neck', 'Shoulders', 'Larynx-PTV', 'Mandible-PTV',
  'OCavity-PTV', 'ParotidCon-PTV', 'ParotidIps-PTV', 'Parotids-PTV', 'PharConst-PTV',
  'Submand-PTV', 'SubmandL-PTV', 'SubmandR-PTV', 'Thyroid-PTV', 'SpinalCord_05',
  'Pituitary', 'OralCavity', 'Submandibular', # additional
]

GDPHMM_PTV = {
  70: 'PTV70',
  63: 'PTV63',
  56: 'PTV56'
}


class StructureSelector:
  """
  Helper class to manage selective loading and indexing of structure components.

  This class solves two problems:
  1. Allows loading only needed structure groups (saves memory/computation)
  2. Provides easy access to structure components without manual index tracking

  Usage:
    # Define what you need
    selector = StructureSelector(groups=['ct', 'oar', 'ptv_sdm'])

    # In dataset __getitem__, load only selected groups
    structure = selector.build_structure(data)

    # Access components easily
    ct = selector.get_component(structure, 'ct')
    oar = selector.get_component(structure, 'oar')

  Available groups:
    - 'ct': CT scan (1 channel)
    - 'ptv': Combined PTV mask (1 channel)
    - 'ptv_masks': Individual PTV masks for 70/63/56 Gy (3 channels)
    - 'oar': OAR binary masks (33 channels)
    - 'oar_sdm': OAR signed distance maps (33 channels)
    - 'ptv_sdm': PTV signed distance map (1 channel)
  """

  # Define all available structure groups and their channel counts
  AVAILABLE_GROUPS = {
    'ct': 1,
    'ptv': 1,
    'ptv_masks': 3,  # ptv70, ptv63, ptv56
    'oar': len(GDPHMM_OAR),  # 33
    'oar_sdm': len(GDPHMM_OAR),  # 33
    'ptv_sdm': 1,
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
      data: Dict with keys 'ct', 'ptv', 'ptv70', 'ptv63', 'ptv56', 'oar', 'oar_sdm', 'ptv_sdm'
            Each value should be a torch.Tensor or numpy array

    Returns:
      structure: (C, H, W) tensor with only selected groups concatenated
    """
    components = []

    for group in self.groups:
      if group == 'ct':
        components.append(data['ct'])
      elif group == 'ptv':
        components.append(data['ptv'])
      elif group == 'ptv_masks':
        components.extend([data['ptv70'], data['ptv63'], data['ptv56']])
      elif group == 'oar':
        components.append(data['oar'])
      elif group == 'oar_sdm':
        components.append(data['oar_sdm'])
      elif group == 'ptv_sdm':
        components.append(data['ptv_sdm'])

    # Concatenate all components
    structure = torch.cat(components, dim=0)
    return structure

  def get_component(self, structure, group_name):
    """
    Extract a specific component from the concatenated structure tensor.

    Args:
      structure: (C, H, W) concatenated structure tensor
      group_name: Name of group to extract (e.g., 'ct', 'oar')

    Returns:
      component: (C_group, H, W) tensor for the requested group
    """
    if group_name not in self._index_map:
      raise ValueError(f"Group '{group_name}' not in selected groups: {self.groups}")

    start, end = self._index_map[group_name]
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
      oar_name: Name of the OAR (from GDPHMM_OAR list)

    Returns:
      Global channel index in the concatenated structure tensor
    """
    if 'oar' not in self._index_map:
      raise ValueError("'oar' group not in selected groups")

    oar_idx = GDPHMM_OAR.index(oar_name)
    oar_start, _ = self._index_map['oar']
    return oar_start + oar_idx

  def get_oar_sdm_index(self, oar_name):
    """
    Get the channel index for a specific OAR SDM within the OAR SDM group.

    Args:
      oar_name: Name of the OAR (from GDPHMM_OAR list)

    Returns:
      Global channel index in the concatenated structure tensor
    """
    if 'oar_sdm' not in self._index_map:
      raise ValueError("'oar_sdm' group not in selected groups")

    oar_idx = GDPHMM_OAR.index(oar_name)
    oar_sdm_start, _ = self._index_map['oar_sdm']
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
      component = self.get_component(structure, group)
      selected.append(component)

    import torch
    return torch.cat(selected, dim=-3)  # Concatenate on channel dim (works for both 3D and 4D)

  def get_ptv(self, structure):
    """
    Get PTV channel from structure tensor.

    Args:
      structure: (B, C, H, W) or (C, H, W) structure tensor

    Returns:
      PTV channel(s): (B, 1, H, W) or (1, H, W)
    """
    return self.get_component(structure, 'ptv')

  def get_ct(self, structure):
    """
    Get CT channel from structure tensor.

    Args:
      structure: (B, C, H, W) or (C, H, W) structure tensor

    Returns:
      CT channel(s): (B, 1, H, W) or (1, H, W)
    """
    return self.get_component(structure, 'ct')

  def get_oar(self, structure, oar_name):
    """
    Get specific OAR channel from structure tensor.

    Args:
      structure: (B, C, H, W) or (C, H, W) structure tensor
      oar_name: Name of the OAR (from GDPHMM_OAR list)

    Returns:
      OAR channel: (B, 1, H, W) or (1, H, W)
    """
    if 'oar' not in self._index_map:
      raise ValueError("'oar' group not in selected groups")

    oar_idx = GDPHMM_OAR.index(oar_name)
    oar_start, _ = self._index_map['oar']
    global_idx = oar_start + oar_idx

    # Handle both batched and unbatched
    if structure.ndim == 4:  # (B, C, H, W)
      return structure[:, global_idx:global_idx+1]
    else:  # (C, H, W)
      return structure[global_idx:global_idx+1]

  def __repr__(self):
    return f"StructureSelector(groups={self.groups}, total_channels={self.total_channels})"


def to_float(x):
  return x.float()

def to_byte(x):
  return x.byte()

def to_plain_tensor(x):
  return torch.as_tensor(x)

def clip_gdphmm_img(x):
  return torch.clip(x, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX)

def normalize_gdphmm_img(x):
  return (x - GDPHMM_IMG_MIN) / (GDPHMM_IMG_MAX - GDPHMM_IMG_MIN)

def normalize_gdphmm_dose(x):
  return (x - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)

def compute_sdm(mask: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
  """
  Compute Signed Distance Map (SDM) for a binary mask.

  SDM provides the minimum distance from each voxel to the contour of the ROI,
  where positive values are inside the ROI, negative values are outside, and
  zero values are on the boundary.

  Args:
      mask: Binary mask (H, W) or (H, W, D) where 1=inside ROI, 0=outside
      voxel_spacing: Physical spacing (is, js) or (is, js, ks) in mm

  Returns:
      SDM with positive values inside, negative outside, 0 on boundary
  """
  mask = mask.astype(bool)

  # Distance transform for points inside (positive distances)
  dist_inside = distance_transform_edt(mask, sampling=voxel_spacing)

  # Distance transform for points outside (will be negated)
  dist_outside = distance_transform_edt(~mask, sampling=voxel_spacing)

  # Combine: positive inside, negative outside
  sdm = dist_inside - dist_outside

  return sdm

class GDPHMMSlicesDataset(Dataset):
  def __init__(
    self,
    dataset_name,
    split,
    img_size,
    cache,
    limit=None,
    skip=0,
    voxel_spacing=(1.0, 1.0, 1.0),
  ):
    self.dataset_name = dataset_name
    self.split = split
    self.img_size = img_size
    self.cache = cache
    self.voxel_spacing = voxel_spacing

    dir = GDPHMM_DATA_BASE_PATH / dataset_name / split
    self.index_map = pd.read_csv(dir / '_index.csv')
    self.index_map['file'] = self.index_map['file'].apply(lambda f: dir / f)
    self.index_map = self.index_map[skip:]
    if limit is not None:
      indices = np.linspace(0, len(self.index_map) - 1, limit, dtype=int)
      self.index_map = self.index_map.iloc[indices]
    self.index_map = self.index_map.to_numpy().tolist()

    self.ct_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(to_float),
      T.Lambda(clip_gdphmm_img),
      T.Lambda(normalize_gdphmm_img),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.oar_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_byte),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])
    self.ptv_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.dose_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(to_float),
      T.Lambda(normalize_gdphmm_dose),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.sdm_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.n_structures = len(GDPHMM_OAR) + 1 + 1
    self.dose_range = (GDPHMM_DOSE_MIN, GDPHMM_DOSE_MAX)

  def __len__(self):
    return len(self.index_map)

  def __getitem__(self, idx):
    file, slice_idx = self.index_map[idx]

    dose_vol, dose_shm = self.cache.get(file.name + ':dose')
    ptv_vol, ptv_shm = self.cache.get(file.name + ':ptv')
    ct_vol, ct_shm = self.cache.get(file.name + ':ct')
    oar_sdm, oar_sdm_shm = self.cache.get(file.name + ':oar_sdm')
    ptv_sdm, ptv_sdm_shm = self.cache.get(file.name + ':ptv_sdm')
    body_vol, body_shm = self.cache.get(file.name + ':body')

    # If any volume is missing, load all from disk and cache them
    if dose_vol is None or ptv_vol is None or ct_vol is None or oar_sdm is None or ptv_sdm is None or body_vol is None:
      vol = np.load(file, allow_pickle=True)['arr_0'].item()

      dose_vol = (vol['dose'] * vol['dose_scale'] * vol['Body'])
      oar_vol = np.stack(
        [vol[name].astype(np.uint8) if name in vol else np.zeros_like(vol['dose'].astype(np.uint8)) for name in GDPHMM_OAR],
        axis=0
      )
      ptv_vol = np.maximum.reduce(
        [vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol]
      )
      ct_vol = vol['img']
      body_vol = vol['Body']
      oar_sdm, ptv_sdm = self.compute_sdms(oar_vol, ptv_vol)

      ct_cache, dose_cache, ptv_cache, body_cache, oar_sdm_cache, ptv_sdm_cache = self.pack_for_cache(
        ct_vol, dose_vol, ptv_vol, body_vol, oar_sdm, ptv_sdm
      )

      self.cache.put(file.name + ':dose', dose_cache)
      self.cache.put(file.name + ':ptv', ptv_cache)
      self.cache.put(file.name + ':ct', ct_cache)
      self.cache.put(file.name + ':oar_sdm', oar_sdm_cache)
      self.cache.put(file.name + ':ptv_sdm', ptv_sdm_cache)
      self.cache.put(file.name + ':body', body_cache)
    else:
      ct_vol, dose_vol, ptv_vol, body_vol, oar_sdm, ptv_sdm = self.unpack_from_cache(
        ct_vol, dose_vol, ptv_vol, body_vol, oar_sdm, ptv_sdm
      )

    oar_vol = (oar_sdm > 0).astype(np.uint8)

    sample = {
      'dose': self.dose_transforms(dose_vol[:, :, slice_idx]),
      'ptv': self.ptv_transforms(ptv_vol[:, :, slice_idx]),
      'ct': self.ct_transforms(ct_vol[:, :, slice_idx]),
      'oar': self.oar_transforms(oar_vol[:, :, :, slice_idx]),
      'body': self.oar_transforms(body_vol[:, :, slice_idx]),
      'oar_sdm': self.sdm_transforms(oar_sdm[:, :, :, slice_idx]),
      'ptv_sdm': self.sdm_transforms(ptv_sdm[:, :, slice_idx]),
      'info': {
        'file_path': str(file),
        'file_name': file.name,
        'volume': file.stem,
        'index': slice_idx
      }
    }

    for shm in [dose_shm, ptv_shm, ct_shm, oar_sdm_shm, ptv_sdm_shm, body_shm]:
      if shm is not None:
        shm.close()

    structure = torch.cat([
      sample['ct'],
      sample['ptv'],
      sample['oar'],
      sample['ptv_sdm'],
      sample['oar_sdm']
    ], dim=0)

    return structure, sample['dose'], sample['body'], sample['info']

  def denormalize_target(self, x):
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

  def finish(self):
    return self.cache.clear()

  def compute_sdms(self, oar_vol, ptv_vol):
    oar_sdm = np.stack([
      compute_sdm(oar_vol[i], self.voxel_spacing)
      for i in range(oar_vol.shape[0])
    ], axis=0)

    ptv_binary = (ptv_vol > 0).astype(np.uint8)
    ptv_sdm = compute_sdm(ptv_binary, self.voxel_spacing)

    return oar_sdm, ptv_sdm

  def pack_for_cache(self, ct_vol, dose_vol, ptv_vol, body_vol, oar_sdm, ptv_sdm):
    ct_pack = np.clip(ct_vol, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX).astype(np.float16)
    dose_norm = (dose_vol - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)
    dose_pack = np.clip(dose_norm * 65535, 0, 65535).astype(np.uint16)
    ptv_pack = np.clip(ptv_vol * 65535, 0, 65535).astype(np.uint16)
    body_pack = body_vol.astype(np.uint8)
    oar_sdm = oar_sdm.astype(np.float16)
    ptv_sdm = ptv_sdm.astype(np.float16)

    return ct_pack, dose_pack, ptv_pack, body_pack, oar_sdm, ptv_sdm

  def unpack_from_cache(self, ct_pack, dose_pack, ptv_pack, body_pack, oar_sdm, ptv_sdm):
    ct_vol = ct_pack.astype(np.float32)
    dose_vol = (dose_pack.astype(np.float32) / 65535.0) * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN
    ptv_vol = ptv_pack.astype(np.float32) / 65535.0
    body_vol = body_pack.astype(np.uint8)
    oar_sdm = oar_sdm.astype(np.float32)
    ptv_sdm = ptv_sdm.astype(np.float32)

    return ct_vol, dose_vol, ptv_vol, body_vol, oar_sdm, ptv_sdm

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
                      Available: ['ct', 'ptv', 'ptv_masks', 'oar', 'oar_sdm', 'ptv_sdm']
                      Example: ['ct', 'oar', 'ptv_sdm'] to load only CT, OARs, and PTV SDM
    limit: Optional limit on number of slices
    skip: Number of slices to skip from start

  Note:
    Assumes slices were preprocessed with --pad-size argument.
    Preprocess your data with:
      python scripts/preprocess_gdphmm_slices.py --pad-size 256 256

  Example usage:
    # Load only CT, OARs, and PTV SDM (saves memory compared to loading everything)
    dataset = GDPHMMPreprocessedSlicesDataset(
      'gdphmm-hn', 'train',
      structure_groups=['ct', 'oar', 'ptv_sdm']
    )
    structure, dose, body, info = dataset[0]

    # Access specific components easily
    ct = dataset.selector.get_component(structure, 'ct')
    oar = dataset.selector.get_component(structure, 'oar')
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
                 Use dataset.selector.get_component(structure, 'ct') to access components
      dose: (1, H, W) tensor, normalized dose
      body: (1, H, W) tensor, body mask
      info: dict with slice metadata
    """
    row = self.index.iloc[idx]
    slice_path = self.data_dir / row['file']

    # Load preprocessed slice (already transformed and padded)
    data_np = np.load(slice_path)

    # Convert all data to tensors with proper type conversions
    data = {
      'ct': torch.from_numpy(data_np['ct']),  # float32, normalized [0, 1]
      'ptv': torch.from_numpy(data_np['ptv']).float() / 100.0,  # uint8 {0, 56, 63, 70} → float32 [0, 1]
      'ptv70': torch.from_numpy(data_np['ptv70']),  # uint8 {0, 1}
      'ptv63': torch.from_numpy(data_np['ptv63']),  # uint8 {0, 1}
      'ptv56': torch.from_numpy(data_np['ptv56']),  # uint8 {0, 1}
      'oar': torch.from_numpy(data_np['oar']),  # uint8 {0, 1}
      'oar_sdm': torch.from_numpy(data_np['oar_sdm']).float(),  # float16 → float32
      'ptv_sdm': torch.from_numpy(data_np['ptv_sdm']).float(),  # float16 → float32
    }

    # Use selector to build structure with only selected groups
    structure = self.selector.build_structure(data)

    # Dose and body are always loaded
    dose = torch.from_numpy(data_np['dose'])  # float32, normalized [0, 1]
    body = torch.from_numpy(data_np['body'])  # uint8 {0, 1}

    # Metadata
    info = {
      'file_path': str(slice_path),
      'file_name': row['file'],
      'volume': row['volume'],
      'index': row['slice_idx']
    }

    return structure, dose, body, info

  def denormalize_target(self, x):
    """Denormalize dose from [0, 1] to [0, 88] Gy."""
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

  def finish(self):
    """Compatibility method (no cleanup needed for preprocessed dataset)."""
    pass


def create_gdphmm_dataset(config, split, structure_groups=None, cache=None):
  """
  Factory function to create GDPHMM dataset based on configuration.

  Args:
    config: Configuration dict
    split: 'train', 'valid', or 'test'
    structure_groups: List of structure groups to load. If None, loads all available groups.
    cache: Cache object (unused, kept for compatibility)

  Config:
    data:
      dataset_name: Name of dataset
  """
  data_config = config['data']
  dataset_name = data_config['dataset_name']
  limit = data_config.get(f'{split}_limit', None)
  skip = data_config.get(f'{split}_skip', 0)

  return GDPHMMPreprocessedSlicesDataset(
    dataset_name=dataset_name,
    split=split,
    structure_groups=structure_groups,
    limit=limit,
    skip=skip,
  )


class PadToFixedSize:
  def __init__(self, target_size, fill=0, padding_mode='constant'):
    """
    Pads an image to the target size.

    Args:
        target_size (int or tuple): Desired output size (H, W). If int, square size is used.
        fill (int or tuple): Pixel fill value for padding. Default 0.
        padding_mode (str): 'constant', 'edge', 'reflect', or 'symmetric'. Default 'constant'.
    """
    if isinstance(target_size, int):
      self.target_size = (target_size, target_size)
    else:
      self.target_size = target_size
    self.fill = fill
    self.padding_mode = padding_mode

  def __call__(self, img):
    """
    Args:
        img (PIL.Image or torch.Tensor): Input image of shape (C,H,W) or (H,W) for grayscale.
    Returns:
        Padded image to self.target_size
    """
    if isinstance(img, torch.Tensor):
      # Ensure shape is (C,H,W)
      if img.ndim == 2:
        img = img.unsqueeze(0)
      h, w = img.shape[-2:]
    elif isinstance(img, Image.Image):
      w, h = img.size
    else:
      raise TypeError(f"Unsupported type {type(img)}")

    pad_h = max(self.target_size[0] - h, 0)
    pad_w = max(self.target_size[1] - w, 0)
    # Pad equally on both sides: (left, top, right, bottom)
    padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)

    y = T.functional.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)
    return y

  def __repr__(self):
    return f"{self.__class__.__name__}(target_size={self.target_size}, fill={self.fill}, padding_mode={self.padding_mode})"


def build_cache(config, manager, split):
  if config[f'{split}_cache_size'] == 'inf':
    return SharedCache(manager)
  elif isinstance(config[f'{split}_cache_size'], str) and config[f'{split}_cache_size'].endswith('B'):
    size_bytes = int(config[f'{split}_cache_size'][:-1])
    return SharedLRUCacheBytes(size_bytes, manager)
  else:
    return SharedLRUCache(config[f'{split}_cache_size'], manager)


class SharedLRUCache:
  """
  Process-safe LRU cache backed by shared memory for heavy numpy arrays.
  Stores key → (SharedMemory.name, shape, dtype, nbytes).
  Evicts least recently used entries when full.
  """

  def __init__(self, max_size, manager):
    self.data = manager.dict()   # key -> (shm_name, shape, dtypei, nbytes)
    self.order = manager.list()  # track LRU
    self.lock = manager.Lock()
    self.max_size = max_size
    self.nbytes = manager.Value('L', 0)  # shared counter (unsigned long)

  def get(self, key):
    with self.lock:
      if key not in self.data:
        return None, None

      shm_name, shape, dtype, _ = self.data[key]

      # touch LRU order
      try:
        self.order.remove(key)
      except ValueError:
        pass
      self.order.append(key)

    # attach outside the lock to reduce contention
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # NOTE: do not close shm here, caller must decide when to release
    return arr, shm

  def put(self, key, value: np.ndarray):
    """
    Stores a numpy array in shared memory.
    If key already exists, overwrite.
    If full, evicts oldest.
    """
    assert isinstance(value, np.ndarray), "SharedLRUCache only supports numpy arrays"

    with self.lock:
      if key in self.data:
        # remove old entry
        shm_name, _, _, nbytes = self.data[key]
        try:
          shm = shared_memory.SharedMemory(name=shm_name)
          shm.close()
          shm.unlink()
        except FileNotFoundError:
          pass
        self.order.remove(key)
        self.nbytes.value -= nbytes

      elif len(self.data) > 0 and len(self.data) >= self.max_size:
        # evict oldest
        oldest = self.order.pop(0)
        shm_name, _, _, nbytes = self.data.pop(oldest)
        try:
          shm = shared_memory.SharedMemory(name=shm_name)
          shm.close()
          shm.unlink()
        except FileNotFoundError:
          pass

        self.nbytes.value -= nbytes

      # allocate new shared block
      shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
      buf = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
      buf[:] = value[:]  # copy data into shared memory

      nbytes = value.nbytes
      self.data[key] = (shm.name, value.shape, value.dtype, nbytes)
      self.order.append(key)
      self.nbytes.value += nbytes

      shm.close()  # keep the block alive, but close handle here

  def __contains__(self, key):
    with self.lock:
      return key in self.data

  def __len__(self):
    with self.lock:
      return len(self.data)

  def keys(self):
    with self.lock:
      return list(self.data.keys())

  def clear(self):
    with self.lock:
      for k, (shm_name, _, _, _) in self.data.items():
        try:
          shm = shared_memory.SharedMemory(name=shm_name)
          shm.close()
          shm.unlink()
        except FileNotFoundError:
          pass

      self.data.clear()
      self.order[:] = []


class SharedCache:
  """
  Process-safe cache backed by shared memory for heavy numpy arrays.
  Stores key → (SharedMemory.name, shape, dtype, nbytes).
  Does NOT evict entries (grows until manually cleared).
  """

  def __init__(self, manager):
    self.data = manager.dict()   # key -> (shm_name, shape, dtype, nbytes)
    self.lock = manager.Lock()
    self.nbytes = manager.Value('L', 0)  # shared counter (unsigned long)

  def get(self, key):
    with self.lock:
      if key not in self.data:
        return None, None

      shm_name, shape, dtype, _ = self.data[key]

    # attach outside lock
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    # caller must close() shm when done
    return arr, shm

  def put(self, key, value: np.ndarray):
    """
    Stores a numpy array in shared memory.
    Overwrites if key already exists.
    """
    assert isinstance(value, np.ndarray), "SharedCache only supports numpy arrays"

    with self.lock:
      if key in self.data:
        # remove old block
        shm_name, _, _, nbytes = self.data[key]
        try:
          shm = shared_memory.SharedMemory(name=shm_name)
          shm.close()
          shm.unlink()
        except FileNotFoundError:
          pass
        self.nbytes.value -= nbytes

      # allocate new block
      shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
      buf = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
      buf[:] = value[:]  # copy data

      nbytes = value.nbytes
      self.data[key] = (shm.name, value.shape, value.dtype, nbytes)
      self.nbytes.value += nbytes

      shm.close()  # keep alive in shared mem

  def __contains__(self, key):
    with self.lock:
      return key in self.data

  def __len__(self):
    with self.lock:
      return len(self.data)

  def keys(self):
    with self.lock:
      return list(self.data.keys())

  def clear(self):
    with self.lock:
      for k, (shm_name, _, _, _) in self.data.items():
        try:
          shm = shared_memory.SharedMemory(name=shm_name)
          shm.close()
          shm.unlink()
        except FileNotFoundError:
          pass

      self.data.clear()
      self.nbytes.value = 0

class SharedLRUCacheBytes:
  """
  Process-safe LRU cache backed by shared memory, evicting by total bytes used.
  Stores key → (SharedMemory.name, shape, dtype, nbytes).
  Evicts least recently used entries until total size <= max_bytes.
  """

  def __init__(self, max_bytes, manager):
    """
    Args:
        max_bytes (int): Maximum total cache size in bytes.
        manager (multiprocessing.Manager): Shared manager for interprocess dict/list.
    """
    self.data = manager.dict()   # key -> (shm_name, shape, dtype, nbytes)
    self.order = manager.list()  # track LRU keys
    self.lock = manager.Lock()
    self.max_bytes = max_bytes
    self.nbytes = manager.Value('L', 0)  # shared counter (unsigned long)

  def get(self, key, wait=True, timeout=5.0, poll_interval=0.01):
    """
    Return (np.ndarray view, SharedMemory handle) or (None, None).

    If the entry is PENDING and wait==True, this will wait up to `timeout`
    seconds (with small sleeps) for the creator to finish and replace the
    PENDING token with the real metadata. If timeout expires, returns (None, None).
    """
    start = time.time()

    while True:
      # Read metadata snapshot
      meta = self.data.get(key, None)
      if meta is None:
        return None, None

      if meta[0] == self.PENDING[0]:
        # somebody reserved it but not completed yet
        if not wait:
          return None, None
        elapsed = time.time() - start
        if elapsed >= timeout:
          return None, None
        time.sleep(poll_interval)
        # retry
        continue

      # meta is real metadata tuple (shm_name, shape, dtype, nbytes)
      shm_name, shape, dtype, _ = meta

      # Update LRU ordering (best-effort). We use a lock because order is a manager list.
      try:
        with self.lock:
          # move to end
          try:
            self.order.remove(key)
          except ValueError:
            pass
          self.order.append(key)
      except Exception:
        # if lock fails, ignore LRU update (non-fatal)
        pass

      # Attach outside of critical section
      try:
        shm = shared_memory.SharedMemory(name=shm_name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        return arr, shm
      except FileNotFoundError:
        # race: creator unlinked before we attached; treat as miss and retry
        return None, None

  def put(self, key, value: np.ndarray, wait_timeout=5.0):
    """
    Atomically create the SharedMemory only once per key using a manager-reservation
    token. If another process created it first, this returns without duplicating memory.

    This function will:
      - set a PENDING token if key not present
      - if set PENDING successfully, create shm and store metadata
      - if someone else already created it, attach and return
    """

    assert isinstance(value, np.ndarray), "SharedLRUCacheBytes only supports numpy arrays"
    v_nbytes = int(value.nbytes)

    # quick reject if too big for cache at all
    if v_nbytes > self.max_bytes:
      return

    # Attempt to atomically reserve the key using setdefault semantics.
    # Only the process that succeeds in inserting the PENDING tuple proceeds to create SHM.
    prev = self.data.setdefault(key, self.PENDING)
    if prev != self.PENDING:
      # Someone else is creating it (prev == PENDING)
      # or it's already done (prev != PENDING)
      # In both cases: **skip immediately**
      return

    # If we get here we either inserted PENDING ourselves (we are creator)
    # or we recycled after a race and now hold the pending slot.
    try:
      # Evict under the lock to maintain nbytes and order consistency
      with self.lock:
        # evict until space
        while self.nbytes.value + v_nbytes > self.max_bytes and len(self.order) > 0:
          oldest = self.order.pop(0)
          meta = self.data.pop(oldest, None)
          if meta and meta[0] != self.PENDING[0]:
            shm_name, _, _, old_nbytes = meta
            try:
              shm_old = shared_memory.SharedMemory(name=shm_name)
              shm_old.close()
              shm_old.unlink()
            except FileNotFoundError:
              pass
            self.nbytes.value -= int(old_nbytes)

        # re-check: maybe someone else wrote this key while we were evicting
        cur = self.data.get(key)
        if cur is not None and cur[0] != self.PENDING[0]:
          # somebody else created it; clean up our PENDING if still present
          return

        # create shared memory block and copy data
        shm = shared_memory.SharedMemory(create=True, size=v_nbytes)
        buf = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
        buf[:] = value[:]

        # store metadata (overwrite PENDING)
        self.data[key] = (shm.name, value.shape, value.dtype, v_nbytes)
        self.order.append(key)
        self.nbytes.value += v_nbytes

        # close local handle; segment remains in /dev/shm and can be attached by others
        shm.close()
    except Exception:
      # on any error remove the PENDING so others can try
      with self.lock:
        meta = self.data.get(key)
        if meta and meta[0] == self.PENDING[0]:
          try:
            del self.data[key]
          except Exception:
            pass
      raise

  def _unlink_shared(self, shm_name):
    """Helper to safely close and unlink a shared memory block."""
    try:
      shm = shared_memory.SharedMemory(name=shm_name)
      shm.close()
      shm.unlink()
    except FileNotFoundError:
      pass
    except FileExistsError:
      pass

  def __contains__(self, key):
    return key in self.data

  def __len__(self):
    return len(self.data)

  def keys(self):
    return list(self.data.keys())

  def clear(self):
    """Remove and unlink all shared memory blocks."""
    with self.lock:
      for k, meta in list(self.data.items()):
        shm_name = meta[0]
        if shm_name and shm_name != self.PENDING[0]:
          self._unlink_shared(shm_name)

      self.data.clear()
      self.order[:] = []
      self.nbytes.value = 0
