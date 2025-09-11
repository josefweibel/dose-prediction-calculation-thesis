from time import time
import torch
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm
import numpy as np

from metrics import DVHScore, DoseScore, HomogeneityIndex
from sampler import GaussianDiffusionSampler

class Evaluator:
  def __init__(self, dataloader, config, device):
    self.dataloader = dataloader
    self.config = config
    self.device = device

    self.metrics = {
      'mse': MeanSquaredError(),
      'rmse': MeanSquaredError(squared=False),
      'psnr': PeakSignalNoiseRatio(data_range=dataloader.dataset.dose_range),
      'ssim': StructuralSimilarityIndexMeasure(data_range=dataloader.dataset.dose_range),
      'ms_ssim': StructuralSimilarityIndexMeasure(data_range=dataloader.dataset.dose_range, multichannel=True),
      'dvh': DVHScore(),
      'dose_score': DoseScore(),
      'homogeneity_index': HomogeneityIndex()
    }

    self.metrics = {k: v.to(device) for k, v in self.metrics.items()}

  def evaluate(self, model, dir=None):
    model.eval()
    sampler = GaussianDiffusionSampler(
      model,
      self.config['beta_1'],
      self.config['beta_T'],
      self.config['T']
    ).to(self.device)

    with torch.no_grad():
      for metric in self.metrics.values():
        metric.reset()

      sum_duration = 0.0
      n_duration = 0

      pbar = tqdm(self.dataloader, desc='Evaluating', leave=False)
      for structure, target, info in pbar:
        structure = structure.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        target = self.dataloader.dataset.denormalize_target(target)

        start = time()
        x_T = torch.randn_like(target, device=self.device)
        pred = sampler(x_T, structure)
        sum_duration += time() - start
        n_duration += target.shape[0]
        pred = torch.clamp(pred, 0., 1.)
        pred = self.dataloader.dataset.denormalize_target(pred)

        for metric in self.metrics.values():
          metric.update(pred, target)

        if dir is not None:
          dir.mkdir(parents=True, exist_ok=True)
          for i, file_info in enumerate(info):
            out_path = dir / file_info['file_name']
            np.save(out_path.with_suffix('.npy'), pred[i].cpu().numpy())

    return {
      'duration': sum_duration / n_duration,
      **{k: m.compute().item() for k, m in self.metrics.items()}
    }

