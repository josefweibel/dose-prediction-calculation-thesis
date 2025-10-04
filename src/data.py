import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import pandas as pd
from multiprocessing import shared_memory, Lock

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

class GDPHMMSlicesDataset(Dataset):
  def __init__(self, dataset_name, split, img_size, cache, limit=None, skip=0):
    self.dataset_name = dataset_name
    self.split = split
    self.img_size = img_size
    self.cache = cache

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

    self.n_structures = len(GDPHMM_OAR) + 1 + 1
    self.dose_range = (GDPHMM_DOSE_MIN, GDPHMM_DOSE_MAX)

  def __len__(self):
    return len(self.index_map)

  def __getitem__(self, idx):
    file, slice_idx = self.index_map[idx]

    dose_vol, dose_shm = self.cache.get(file.name + ':dose')
    if dose_vol is None:
      vol = np.load(file, allow_pickle=True)['arr_0'].item()

      dose_vol = (vol['dose'] * vol['dose_scale'] * vol['Body']).astype(np.float32)
      oar_vol = np.stack(
        [vol[name].astype(np.uint8) if name in vol else np.zeros_like(vol['dose'].astype(np.uint8)) for name in GDPHMM_OAR],
        axis=0
      ).astype(np.uint8)
      ptv_vol = np.maximum.reduce(
        [vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol]
      ).astype(np.float32)
      ct_vol = vol['img'].astype(np.float32)

      self.cache.put(file.name + ':dose', dose_vol)
      self.cache.put(file.name + ':ptv', ptv_vol)
      self.cache.put(file.name + ':ct', ct_vol)
      self.cache.put(file.name + ':oar', oar_vol)
    else:
      ptv_vol, ptv_shm = self.cache.get(file.name + ':ptv')
      ct_vol, ct_shm = self.cache.get(file.name + ':ct')
      oar_vol, oar_shm = self.cache.get(file.name + ':oar')

    sample = {
      'dose': self.dose_transforms(dose_vol[:, :, slice_idx]),
      'ptv': self.ptv_transforms(ptv_vol[:, :, slice_idx]),
      'ct': self.ct_transforms(ct_vol[:, :, slice_idx]),
      'oar': self.oar_transforms(oar_vol[:, :, :, slice_idx]),
      'info': {
        'file_path': str(file),
        'file_name': file.name,
        'volume': file.stem,
        'index': slice_idx
      }
    }

    if dose_shm:
      dose_shm.close()
      ptv_shm.close()
      ct_shm.close()
      oar_shm.close()

    structure = torch.cat([sample['ct'], sample['ptv'], sample['oar']], dim=0)
    return structure, sample['dose'], sample['info']

  def denormalize_target(self, x):
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

  def finish(self):
    return self.cache.clear()

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


class SharedLRUCache:
  """
  Process-safe LRU cache backed by shared memory for heavy numpy arrays.
  Stores key → (SharedMemory.name, shape, dtype, nbytes).
  Evicts least recently used entries when full.
  """

  def __init__(self, max_size, manager):
    self.data = manager.dict()   # key -> (shm_name, shape, dtypei, nbytes)
    self.order = manager.list()  # track LRU
    self.lock = Lock()
    self.max_size = max_size
    self.nbytes = 0

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
        self.nbytes -= nbytes

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

        self.nbytes -= nbytes

      # allocate new shared block
      shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
      buf = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
      buf[:] = value[:]  # copy data into shared memory

      nbytes = value.nbytes
      self.data[key] = (shm.name, value.shape, value.dtype, nbytes)
      self.order.append(key)
      self.nbytes += nbytes

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
    self.lock = Lock()
    self.nbytes = 0

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
        self.nbytes -= nbytes

      # allocate new block
      shm = shared_memory.SharedMemory(create=True, size=value.nbytes)
      buf = np.ndarray(value.shape, dtype=value.dtype, buffer=shm.buf)
      buf[:] = value[:]  # copy data

      nbytes = value.nbytes
      self.data[key] = (shm.name, value.shape, value.dtype, nbytes)
      self.nbytes += nbytes

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
      self.nbytes = 0

