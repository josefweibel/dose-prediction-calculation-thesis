import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

GDPHMM_DATA_PATH = Path(__file__).parents[1] / 'data' / 'gdphmm-hn'

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


class GDPHMMSlicesDataset(Dataset):
  def __init__(self, split, img_size, limit=None):
    self.split = split
    self.img_size = img_size

    self.ct_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(lambda x: x.float()),  # Convert to float32
      T.Lambda(lambda x: torch.clip(x, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX)),
      T.Lambda(lambda x: (x - GDPHMM_IMG_MIN) / (GDPHMM_IMG_MAX - GDPHMM_IMG_MIN)),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.oar_transforms = T.Compose([
      T.Lambda(lambda x: torch.as_tensor(x)),
      T.Lambda(lambda x: x.byte()),  # Convert to unit8
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])
    self.ptv_transforms = self.oar_transforms

    self.dose_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(lambda x: x.float()),  # Convert to float32
      T.Lambda(lambda x: (x - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    data = self.__load_data(split, limit)
    data = self.__transform(data)
    self.data = self.__prepare_structure(data)
    self.keys = sorted(data.keys())
    self.n_structures = len(GDPHMM_OAR) + 1 + 1

  def __len__(self):
    return len(self.keys)

  def __getitem__(self, idx):
    key = self.keys[idx]
    sample = self.data[key]

    return sample['structure'], sample['target'], sample['info']

  def __load_data(self, split, limit):
    data = {}

    files = list((GDPHMM_DATA_PATH / split).glob('*.npz'))
    for file in tqdm(files, desc=f'Loading Dataset {split}'):
      vol = np.load(file, allow_pickle=True)['arr_0'].item()
      dose_vol = vol['dose'] * vol['dose_scale'] * vol['Body']

      oar_vol = np.stack([vol[name] if name in vol else np.zeros_like(vol['dose']) for name in GDPHMM_OAR], axis=0)
      ptv_vol = np.maximum.reduce([vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol])

      for idx in range(vol['dose'].shape[2]):
        data[f'{file.stem}_{idx}'] = {
          'dose': dose_vol[:, :, idx],
          'ptv': ptv_vol[:, :, idx],
          'oar': oar_vol[:, :, :, idx],
          'ct': vol['img'][:, :, idx],
          'info': {
            'file_path': str(file),
            'file_name': file.name,
            'volume': file.stem,
            'index': idx
          }
        }

        if limit is not None and len(data) >= limit:
          return data

    return data

  def __prepare_structure(self, data):
    return {
      k: {
        'target': v['dose'],
        'info': v['info'],
        'structure': np.concatenate([v['ct'], v['ptv'], v['oar']], axis=0)
      }
      for k, v in data.items()
    }

  def __transform(self, data):
    return {
      k: {
        'dose': self.dose_transforms(v['dose']),
        'ptv': self.ptv_transforms(v['ptv']),
        'oar': self.oar_transforms(v['oar']),
        'ct': self.ct_transforms(v['ct']),
        'info': v['info'],
      }
      for k, v in data.items()
    }

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
