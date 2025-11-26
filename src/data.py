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
    include_sdm=False,
    voxel_spacing=(1.0, 1.0, 1.0),
  ):
    self.dataset_name = dataset_name
    self.split = split
    self.img_size = img_size
    self.cache = cache
    self.include_sdm = include_sdm
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
    oar_vol, oar_shm = self.cache.get(file.name + ':oar')
    body_vol, body_shm = self.cache.get(file.name + ':body')

    # If any volume is missing, load all from disk and cache them
    if dose_vol is None or ptv_vol is None or ct_vol is None or oar_vol is None or body_vol is None:
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

      ct_cache, dose_cache, ptv_cache, oar_cache, body_cache = self.pack_for_cache(
        ct_vol, dose_vol, ptv_vol, oar_vol, body_vol
      )

      self.cache.put(file.name + ':dose', dose_cache)
      self.cache.put(file.name + ':ptv', ptv_cache)
      self.cache.put(file.name + ':ct', ct_cache)
      self.cache.put(file.name + ':oar', oar_cache)
      self.cache.put(file.name + ':body', body_cache)
    else:
      ct_vol, dose_vol, ptv_vol, oar_vol, body_vol = self.unpack_from_cache(
        ct_vol, dose_vol, ptv_vol, oar_vol, body_vol
      )

    sample = {
      'dose': self.dose_transforms(dose_vol[:, :, slice_idx]),
      'ptv': self.ptv_transforms(ptv_vol[:, :, slice_idx]),
      'ct': self.ct_transforms(ct_vol[:, :, slice_idx]),
      'oar': self.oar_transforms(oar_vol[:, :, :, slice_idx]),
      'body': self.oar_transforms(body_vol[:, :, slice_idx]),
      'info': {
        'file_path': str(file),
        'file_name': file.name,
        'volume': file.stem,
        'index': slice_idx
      }
    }

    if self.include_sdm:
      oar_sdm_vol, oar_sdm_shm = self.cache.get(file.name + ':oar_sdm')
      ptv_sdm_vol, ptv_sdm_shm = self.cache.get(file.name + ':ptv_sdm')

      if oar_sdm_vol is None or ptv_sdm_vol is None:
        # Compute SDM for each OAR channel on full 3D volume
        oar_sdm_vol = np.stack([
          compute_sdm(oar_vol[i], self.voxel_spacing)
          for i in range(oar_vol.shape[0])
        ], axis=0)

        # Compute SDM for PTV on full 3D volume
        ptv_binary = (ptv_vol > 0).astype(np.uint8)
        ptv_sdm_vol = compute_sdm(ptv_binary, self.voxel_spacing)

        # Cache the computed SDMs
        oar_sdm_cache, ptv_sdm_cache = self.pack_sdm_for_cache(oar_sdm_vol, ptv_sdm_vol)
        self.cache.put(file.name + ':oar_sdm', oar_sdm_cache)
        self.cache.put(file.name + ':ptv_sdm', ptv_sdm_cache)
      else:
        oar_sdm_vol, ptv_sdm_vol = self.unpack_sdm_from_cache(oar_sdm_vol, ptv_sdm_vol)

      sample['oar_sdm'] = self.sdm_transforms(oar_sdm_vol[:, :, :, slice_idx])
      sample['ptv_sdm'] = self.sdm_transforms(ptv_sdm_vol[:, :, slice_idx])

      # Close shared memory handles
      if oar_sdm_shm is not None:
        oar_sdm_shm.close()
      if ptv_sdm_shm is not None:
        ptv_sdm_shm.close()

    for shm in [dose_shm, ptv_shm, ct_shm, oar_shm, body_shm]:
      if shm is not None:
        shm.close()

    # Build structure input - always return all available data
    # The model/trainer will decide which channels to use based on config
    if self.include_sdm:
      structure = torch.cat([
        sample['ct'],
        sample['ptv'],
        sample['oar'],
        sample['ptv_sdm'].unsqueeze(0),
        sample['oar_sdm']
      ], dim=0)
    else:
      structure = torch.cat([sample['ct'], sample['ptv'], sample['oar']], dim=0)

    return structure, sample['dose'], sample['body'], sample['info']

  def denormalize_target(self, x):
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

  def finish(self):
    return self.cache.clear()

  def pack_for_cache(self, ct_vol, dose_vol, ptv_vol, oar_vol, body_vol):
    ct_pack = np.clip(ct_vol, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX).astype(np.float16)
    dose_norm = (dose_vol - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)
    dose_pack = np.clip(dose_norm * 65535, 0, 65535).astype(np.uint16)
    ptv_pack = np.clip(ptv_vol * 65535, 0, 65535).astype(np.uint16)
    oar_pack = oar_vol.astype(np.uint8)
    body_pack = body_vol.astype(np.uint8)
    return ct_pack, dose_pack, ptv_pack, oar_pack, body_pack

  def unpack_from_cache(self, ct_pack, dose_pack, ptv_pack, oar_pack, body_pack):
    ct_vol = ct_pack.astype(np.float32)
    dose_vol = (dose_pack.astype(np.float32) / 65535.0) * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN
    ptv_vol = ptv_pack.astype(np.float32) / 65535.0
    oar_vol = oar_pack.astype(np.uint8)
    body_vol = body_pack.astype(np.uint8)
    return ct_vol, dose_vol, ptv_vol, oar_vol, body_vol

  def pack_sdm_for_cache(self, oar_sdm_vol, ptv_sdm_vol):
    """Pack SDM volumes for efficient caching using float16."""
    oar_sdm_pack = oar_sdm_vol.astype(np.float16)
    ptv_sdm_pack = ptv_sdm_vol.astype(np.float16)
    return oar_sdm_pack, ptv_sdm_pack

  def unpack_sdm_from_cache(self, oar_sdm_pack, ptv_sdm_pack):
    """Unpack SDM volumes from cache."""
    oar_sdm_vol = oar_sdm_pack.astype(np.float32)
    ptv_sdm_vol = ptv_sdm_pack.astype(np.float32)
    return oar_sdm_vol, ptv_sdm_vol

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
