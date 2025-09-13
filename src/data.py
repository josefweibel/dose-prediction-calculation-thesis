import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
import pandas as pd

GDPHMM_DATA_BASE_PATH = Path(__file__).parents[1] / 'data'

GDPHMM_IMG_MIN = -1024
GDPHMM_IMG_MAX = 3071

GDPHMM_DOSE_MIN = 0
GDPHMM_DOSE_MAX = 122

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
  def __init__(self, dataset_name, split, img_size, limit=None):
    self.dataset_name = dataset_name
    self.split = split
    self.img_size = img_size

    dir = GDPHMM_DATA_BASE_PATH / dataset_name / split
    self.index_map = pd.read_csv(dir / '_index.csv')
    self.index_map['file'] = self.index_map['file'].apply(lambda f: dir / f)
    if limit is not None:
      self.index_map = self.index_map.iloc[:limit]
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
    self.ptv_transforms = self.oar_transforms

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
    vol = np.load(file, allow_pickle=True)['arr_0'].item()

    # reconstruct relevant slices
    dose_vol = vol['dose'] * vol['dose_scale'] * vol['Body']
    oar_vol = np.stack(
      [vol[name] if name in vol else np.zeros_like(vol['dose']) for name in GDPHMM_OAR],
      axis=0
    )
    ptv_vol = np.maximum.reduce(
      [vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol]
    )

    sample = {
      'dose': self.dose_transforms(dose_vol[:, :, slice_idx]),
      'ptv': self.ptv_transforms(ptv_vol[:, :, slice_idx]),
      'oar': self.oar_transforms(oar_vol[:, :, :, slice_idx]),
      'ct': self.ct_transforms(vol['img'][:, :, slice_idx]),
      'info': {
        'file_path': str(file),
        'file_name': file.name,
        'volume': file.stem,
        'index': slice_idx
      }
    }

    structure = torch.cat([sample['ct'], sample['ptv'], sample['oar']], dim=0)
    return structure, sample['dose'], sample['info']

  def denormalize_target(self, x):
    return x * (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN) + GDPHMM_DOSE_MIN

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
