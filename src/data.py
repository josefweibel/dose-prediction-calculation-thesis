import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm

GDPHMM_DATA_PATH = Path(__file__).parents[1] / 'data' / 'gdphmm-hn'

class GDPHMMSlicesDataset(Dataset):
  def __init__(self, split, img_size, structures=None, limit=None):
    self.split = split
    self.img_size = img_size

    self.img_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.mask_transforms = T.Compose([
      T.ToTensor(),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    self.target_transforms = T.Compose([
      T.ToTensor(),
      T.Lambda(lambda x: (x - torch.min(x)) / (torch.max(x) - torch.min(x))),
      PadToFixedSize(img_size, fill=0, padding_mode='constant'),
    ])

    data = self.__load_data(split, limit)
    data = self.__transform(data)

    self.keys = sorted(data.keys())

    self.structures = structures
    if structures is None:
      self.structures = ['img'] + sorted(np.unique([s['masks'] for s in data.values()]))

    self.data = self.__prepare_structure(data, self.structures)

  def __len__(self):
    return len(self.keys)

  def __getitem__(self, idx):
    key = self.keys[idx]
    sample = self.data[key]

    return sample['structure'], sample['target'], sample['info']

  def __load_data(self, split, limit):
    data = {}

    files = list((GDPHMM_DATA_PATH / split).glob('*.npz'))
    if limit is not None:
      files = sorted(files)[:limit]

    for file in tqdm(files, desc=f'Loading Dataset {split}'):
      volume = np.load(file, allow_pickle=True)['arr_0'].item()

      for idx in range(volume['dose'].shape[2]):
        print('target', volume['dose'][:, :, idx])
        data[f'{file.stem}_{idx}'] = {
          'target': volume['dose'][:, :, idx],
          'masks': volume['all_mask'],
          'img': volume['img'][:, :, idx],
          'structures': {
            k: volume[k][:, :, idx] for k in volume['all_mask'] if k in volume
          },
          'info': {
            'file_path': str(file),
            'file_name': file.name,
            'volume': file.stem,
            'index': idx
          }
        }

    return data

  def __prepare_structure(self, data, structures):
    assert structures[0] == 'img', "First structure must be 'img'"

    return {
      k: {
        'target': v['target'],
        'info': v['info'],
        'structure': np.stack([v['img'], *[v['structures'].get(s, np.zeros_like(v['target'])) for s in structures[1:]]], axis=0)
      }
      for k, v in data.items()
    }

  def __transform(self, data):
    return {
      k: {
        'target': self.target_transforms(v['target']),
        'masks': v['masks'],
        'img': self.img_transforms(v['img']),
        'structure': {s: self.mask_transforms(v['structure'][s]) for s in v['structure'].keys()},
        'info': v['info'],
      }
      for k, v in data.items()
    }


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

    return T.functional.pad(img, padding, fill=self.fill, padding_mode=self.padding_mode)

  def __repr__(self):
    return f"{self.__class__.__name__}(target_size={self.target_size}, fill={self.fill}, padding_mode={self.padding_mode})"
