import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from multiprocessing import shared_memory
import time
from scipy.ndimage import distance_transform_edt, zoom

from data import StructureSelector

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

# Version 2: Remove last three values (Pituitary, OralCavity, Submandibular)
GDPHMM_OAR_V2 = GDPHMM_OAR[:-3]

GDPHMM_PTV = {
  70: 'PTV70',
  63: 'PTV63',
  56: 'PTV56'
}


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


def process_angle_plate(angle_plate_2d, img_shape, isocenter, z_range=5):
  """
  Convert 2D angle plate to 3D volume around the isocenter.

  Based on GDP-HMM implementation: creates a 3D volume by replicating the 2D binary angle plate
  within a range around the isocenter (±z_range slices).

  Args:
    angle_plate_2d: (H, W) 2D binary angle plate array
    img_shape: (H, W, D) shape of the target volume
    isocenter: (z, y, x) coordinates of the isocenter
    z_range: number of slices above and below isocenter to fill (default: 5)

  Returns:
    angle_plate_3d: (H, W, D) 3D binary volume with angle plate around isocenter
  """
  # Initialize empty 3D volume
  angle_plate_3d = np.zeros(img_shape, dtype=np.uint8)

  # Determine z-range around isocenter
  z_begin = max(0, int(isocenter[0]) - z_range)
  z_end = min(img_shape[2], int(isocenter[0]) + z_range)

  # Resize angle_plate to match img_shape if needed (order=0 preserves binary values)
  if angle_plate_2d.shape[0] != img_shape[0] or angle_plate_2d.shape[1] != img_shape[1]:
    zoom_factors = (img_shape[0] / angle_plate_2d.shape[0],
                    img_shape[1] / angle_plate_2d.shape[1])
    angle_plate_2d = zoom(angle_plate_2d, zoom_factors, order=0)

  # Replicate the 2D plate across the z-range
  for z in range(z_begin, z_end):
    angle_plate_3d[:, :, z] = angle_plate_2d

  return angle_plate_3d


class GDPHMMSlicesDataset(Dataset):
  def __init__(
    self,
    dataset_name,
    split,
    img_size,
    cache,
    structure_groups=None,
    limit=None,
    skip=0,
    voxel_spacing=(1.0, 1.0, 1.0),
    mode='2d',
  ):
    self.dataset_name = dataset_name
    self.split = split
    self.img_size = img_size
    self.cache = cache
    self.voxel_spacing = voxel_spacing
    self.mode = mode

    self.selector = StructureSelector(groups=structure_groups)

    dir = GDPHMM_DATA_BASE_PATH / dataset_name / split
    self.index_map = pd.read_csv(dir / '_index.csv')
    self.index_map['file'] = self.index_map['file'].apply(lambda f: dir / f)

    if mode == '3d':
      self.index_map = self.index_map.drop_duplicates('file').reset_index(drop=True)

    self.index_map = self.index_map[skip:]
    if limit is not None:
      indices = np.linspace(0, len(self.index_map) - 1, limit, dtype=int)
      self.index_map = self.index_map.iloc[indices]
    self.index_map = self.index_map.to_numpy().tolist()

    self.ct_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      T.Lambda(clip_gdphmm_img),
      T.Lambda(normalize_gdphmm_img),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.oar_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_byte),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])
    self.ptv_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.dose_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      T.Lambda(normalize_gdphmm_dose),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.sdm_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.angle_plate_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_byte),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.beam_plate_transforms = T.Compose([
      T.Lambda(to_plain_tensor),
      T.Lambda(to_float),
      PadToFixedSize(img_size, fill=0, padding_mode='constant', mode=mode),
    ])

    self.n_structures = self.selector.total_channels
    self.dose_range = (GDPHMM_DOSE_MIN, GDPHMM_DOSE_MAX)

  def __len__(self):
    return len(self.index_map)

  def __getitem__(self, idx):
    file, slice_idx = self.index_map[idx]

    dose_vol, dose_shm = self.cache.get(file.name + ':dose')
    ptv_vol, ptv_shm = self.cache.get(file.name + ':ptv')
    ptv70_vol, ptv70_shm = self.cache.get(file.name + ':ptv70')
    ptv63_vol, ptv63_shm = self.cache.get(file.name + ':ptv63')
    ptv56_vol, ptv56_shm = self.cache.get(file.name + ':ptv56')
    ct_vol, ct_shm = self.cache.get(file.name + ':ct')
    oar_sdm, oar_sdm_shm = self.cache.get(file.name + ':oar_sdm')
    ptv_sdm, ptv_sdm_shm = self.cache.get(file.name + ':ptv_sdm')
    body_vol, body_shm = self.cache.get(file.name + ':body')
    beam_plate_vol, beam_plate_shm = self.cache.get(file.name + ':beam_plate')
    angle_plate_vol, angle_plate_shm = self.cache.get(file.name + ':angle_plate')

    # If any volume is missing, load all from disk and cache them
    if (dose_vol is None or ptv_vol is None or ptv70_vol is None or ptv63_vol is None or
        ptv56_vol is None or ct_vol is None or oar_sdm is None or ptv_sdm is None or body_vol is None or
        beam_plate_vol is None or angle_plate_vol is None):
      vol = np.load(file, allow_pickle=True)['arr_0'].item()

      dose_vol = (vol['dose'] * vol['dose_scale'] * vol['Body'])
      oar_vol = np.stack(
        [vol[name].astype(np.uint8) if name in vol else np.zeros_like(vol['dose'].astype(np.uint8)) for name in GDPHMM_OAR],
        axis=0
      )

      # Create individual PTV masks
      ptv70_vol = vol['PTV70'].astype(np.uint8) if 'PTV70' in vol else np.zeros_like(vol['dose'].astype(np.uint8))
      ptv63_vol = vol['PTV63'].astype(np.uint8) if 'PTV63' in vol else np.zeros_like(vol['dose'].astype(np.uint8))
      ptv56_vol = vol['PTV56'].astype(np.uint8) if 'PTV56' in vol else np.zeros_like(vol['dose'].astype(np.uint8))

      # Combined PTV with dose levels
      ptv_vol = np.maximum.reduce(
        [vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol]
      )
      ct_vol = vol['img']
      body_vol = vol['Body']
      oar_sdm, ptv_sdm = self.compute_sdms(oar_vol, ptv_vol)

      # Process beam_plate (already 3D) and angle_plate (2D to 3D)
      beam_plate_vol = vol['beam_plate']
      angle_plate_2d = vol['angle_plate']
      isocenter = vol['isocenter']
      angle_plate_vol = process_angle_plate(angle_plate_2d, ct_vol.shape, isocenter, z_range=5)

      ct_cache, dose_cache, ptv_cache, ptv70_cache, ptv63_cache, ptv56_cache, body_cache, oar_sdm_cache, ptv_sdm_cache, beam_plate_cache, angle_plate_cache = self.pack_for_cache(
        ct_vol, dose_vol, ptv_vol, ptv70_vol, ptv63_vol, ptv56_vol, body_vol, oar_sdm, ptv_sdm, beam_plate_vol, angle_plate_vol
      )

      self.cache.put(file.name + ':dose', dose_cache)
      self.cache.put(file.name + ':ptv', ptv_cache)
      self.cache.put(file.name + ':ptv70', ptv70_cache)
      self.cache.put(file.name + ':ptv63', ptv63_cache)
      self.cache.put(file.name + ':ptv56', ptv56_cache)
      self.cache.put(file.name + ':ct', ct_cache)
      self.cache.put(file.name + ':oar_sdm', oar_sdm_cache)
      self.cache.put(file.name + ':ptv_sdm', ptv_sdm_cache)
      self.cache.put(file.name + ':body', body_cache)
      self.cache.put(file.name + ':beam_plate', beam_plate_cache)
      self.cache.put(file.name + ':angle_plate', angle_plate_cache)
    else:
      ct_vol, dose_vol, ptv_vol, ptv70_vol, ptv63_vol, ptv56_vol, body_vol, oar_sdm, ptv_sdm, beam_plate_vol, angle_plate_vol = self.unpack_from_cache(
        ct_vol, dose_vol, ptv_vol, ptv70_vol, ptv63_vol, ptv56_vol, body_vol, oar_sdm, ptv_sdm, beam_plate_vol, angle_plate_vol
      )

    oar_vol = (oar_sdm > 0).astype(np.uint8)

    if self.mode == '2d':
      sample = {
        'dose': self.dose_transforms(dose_vol[:, :, slice_idx]),
        'ptv': self.ptv_transforms(ptv_vol[:, :, slice_idx]),
        'ptv70': self.oar_transforms(ptv70_vol[:, :, slice_idx]),
        'ptv63': self.oar_transforms(ptv63_vol[:, :, slice_idx]),
        'ptv56': self.oar_transforms(ptv56_vol[:, :, slice_idx]),
        'ct': self.ct_transforms(ct_vol[:, :, slice_idx]),
        'oar': self.oar_transforms(oar_vol[:, :, :, slice_idx]),
        'body': self.oar_transforms(body_vol[:, :, slice_idx]),
        'oar_sdm': self.sdm_transforms(oar_sdm[:, :, :, slice_idx]),
        'ptv_sdm': self.sdm_transforms(ptv_sdm[:, :, slice_idx]),
        'beam_plate': self.beam_plate_transforms(beam_plate_vol[:, :, slice_idx]),
        'angle_plate': self.angle_plate_transforms(angle_plate_vol[:, :, slice_idx]),
        'info': {
          'file_path': str(file),
          'file_name': file.name,
          'volume': file.stem,
          'index': slice_idx
        }
      }
    else:
      sample = {
        'dose': self.dose_transforms(dose_vol),
        'ptv': self.ptv_transforms(ptv_vol),
        'ptv70': self.oar_transforms(ptv70_vol),
        'ptv63': self.oar_transforms(ptv63_vol),
        'ptv56': self.oar_transforms(ptv56_vol),
        'ct': self.ct_transforms(ct_vol),
        'oar': self.oar_transforms(oar_vol),
        'body': self.oar_transforms(body_vol),
        'oar_sdm': self.sdm_transforms(oar_sdm),
        'ptv_sdm': self.sdm_transforms(ptv_sdm),
        'beam_plate': self.beam_plate_transforms(beam_plate_vol),
        'angle_plate': self.angle_plate_transforms(angle_plate_vol),
        'info': {
          'file_path': str(file),
          'file_name': file.name,
          'volume': file.stem,
        }
      }

    for shm in [dose_shm, ptv_shm, ptv70_shm, ptv63_shm, ptv56_shm, ct_shm, oar_sdm_shm, ptv_sdm_shm, body_shm, beam_plate_shm, angle_plate_shm]:
      if shm is not None:
        shm.close()

    structure = self.selector.build_structure(sample)

    return structure, sample['dose'], sample['info']

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

  def pack_for_cache(self, ct_vol, dose_vol, ptv_vol, ptv70_vol, ptv63_vol, ptv56_vol, body_vol, oar_sdm, ptv_sdm, beam_plate_vol, angle_plate_vol):
    ct_pack = np.clip(ct_vol, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX).astype(np.float16)
    dose_norm = (dose_vol - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)
    dose_pack = np.clip(dose_norm * 65535, 0, 65535).astype(np.uint16)
    ptv_pack = np.clip(ptv_vol * 65535, 0, 65535).astype(np.uint16)
    ptv70_pack = ptv70_vol.astype(np.uint8)
    ptv63_pack = ptv63_vol.astype(np.uint8)
    ptv56_pack = ptv56_vol.astype(np.uint8)
    body_pack = body_vol.astype(np.uint8)
    oar_sdm = oar_sdm.astype(np.float16)
    ptv_sdm = ptv_sdm.astype(np.float16)
    beam_plate_pack = beam_plate_vol.astype(np.float32)
    angle_plate_pack = angle_plate_vol.astype(np.uint8)

    return ct_pack, dose_pack, ptv_pack, ptv70_pack, ptv63_pack, ptv56_pack, body_pack, oar_sdm, ptv_sdm, beam_plate_pack, angle_plate_pack

  def unpack_from_cache(self, ct_pack, dose_pack, ptv_pack, ptv70_pack, ptv63_pack, ptv56_pack, body_pack, oar_sdm, ptv_sdm, beam_plate_pack, angle_plate_pack):
    ct_vol = ct_pack.astype(np.float32)
    dose_vol = (dose_pack.astype(np.float32) / 65535.0) * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN
    ptv_vol = ptv_pack.astype(np.float32) / 65535.0
    ptv70_vol = ptv70_pack.astype(np.uint8)
    ptv63_vol = ptv63_pack.astype(np.uint8)
    ptv56_vol = ptv56_pack.astype(np.uint8)
    body_vol = body_pack.astype(np.uint8)
    oar_sdm = oar_sdm.astype(np.float32)
    ptv_sdm = ptv_sdm.astype(np.float32)
    beam_plate_vol = beam_plate_pack.astype(np.float32)
    angle_plate_vol = angle_plate_pack.astype(np.uint8)

    return ct_vol, dose_vol, ptv_vol, ptv70_vol, ptv63_vol, ptv56_vol, body_vol, oar_sdm, ptv_sdm, beam_plate_vol, angle_plate_vol


class PadToFixedSize:
  def __init__(self, target_size, fill=0, padding_mode='constant', mode='2d'):
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
    self.mode = mode

  def __call__(self, img):
    """
    Args:
        img (PIL.Image or torch.Tensor): Input image of shape (C,H,W), (H,W) for grayscale, or (C,H,W,D) for 4D volumes.
    Returns:
        Padded image to self.target_size. For 4D volumes, only H and W dimensions are padded (last D dimension is not padded).
    """
    if self.mode == '2d':
      if img.ndim == 2:
        h, w = img.shape
        img = img.unsqueeze(0)
      elif img.ndim == 3:
        _, h, w = img.shape
    elif self.mode == '3d':
      if img.ndim == 3:
        h, w, _ = img.shape
        img = img.unsqueeze(0)
      elif img.ndim == 4:
        _, h, w, _ = img.shape

    pad_h = max(self.target_size[0] - h, 0)
    pad_w = max(self.target_size[1] - w, 0)
    # Pad equally on both sides: (left, right, top, bottom)
    padding = (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2)

    if self.mode == '3d':
      padding = (0, 0) + padding  # (D_left, D_right, W_left, W_right, H_left, H_right)

    y = torch.nn.functional.pad(img, padding, value=self.fill, mode=self.padding_mode)
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

class LRUCache:
  """
  Simple cache for numpy arrays stored in local memory (not shared).
  No size limit - stores all entries until manually cleared.
  Each instance maintains its own isolated cache.
  """

  def __init__(self):
    self.data = {}

  def get(self, key):
    """
    Return (np.ndarray, None) or (None, None).
    Second return value is None for compatibility with shared cache interface.
    """
    if key not in self.data:
      return None, None
    return self.data[key], None

  def put(self, key, value: np.ndarray):
    """Store a numpy array in local memory."""
    assert isinstance(value, np.ndarray), "LRUCacheBytes only supports numpy arrays"
    self.data[key] = value.copy()

  def __contains__(self, key):
    return key in self.data

  def __len__(self):
    return len(self.data)

  def keys(self):
    return list(self.data.keys())

  def clear(self):
    """Remove all entries."""
    self.data.clear()


class SharedLRUCacheBytes:
  """
  Process-safe LRU cache backed by shared memory, evicting by total bytes used.
  Stores key → (SharedMemory.name, shape, dtype, nbytes).
  Evicts least recently used entries until total size <= max_bytes.
  """

  PENDING = ("__PENDING__", None, None, 0)

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
