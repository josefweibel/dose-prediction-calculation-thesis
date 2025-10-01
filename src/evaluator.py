from time import time
import torch
import torch.distributed as dist
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import numpy as np
from pathlib import Path

from metrics import DVHScore, DoseScore, HomogeneityIndex
from sampler import build_sampler

class Evaluator:
  def __init__(self, dataloader, model_config, sampling_config, device, rank=0, world_size=1):
    self.dataloader = dataloader
    self.model_config = model_config
    self.sampling_config = sampling_config
    self.device = device
    self.rank = rank
    self.world_size = world_size

    dose_range = dataloader.dataset.dose_range[1] - dataloader.dataset.dose_range[0]
    self.metrics = {
      'mse': MeanSquaredError(),
      'rmse': MeanSquaredError(squared=False),
      'psnr': PeakSignalNoiseRatio(data_range=dose_range),
      'ssim': StructuralSimilarityIndexMeasure(data_range=dose_range),
      'ms_ssim': MultiScaleStructuralSimilarityIndexMeasure(
        data_range=dataloader.dataset.dose_range,
        betas=(0.0448, 0.2856, 0.3001, 0.2363), # only 4 values since image size is 160x176
      ),
      'dvh': DVHScore(),
      'dose_score': DoseScore(),
      'homogeneity_index': HomogeneityIndex()
    }

    self.metrics = {k: v.to(device) for k, v in self.metrics.items()}

  def evaluate(self, model, dir=None):
    model.eval()
    sampler = build_sampler(self.model_config, self.sampling_config, model).to(self.device)

    with torch.no_grad():
      for metric in self.metrics.values():
        metric.reset()

      sum_duration = 0.0
      n_duration = 0

      pbar = tqdm(self.dataloader, desc='Evaluating', leave=False, disable=self.rank != 0)
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
          if self.rank == 0:
            dir.mkdir(parents=True, exist_ok=True)

          dist.barrier()

          for i, (file_name, slice_index) in enumerate(zip(info['file_name'], info['index'])):
            p = Path(file_name)
            out_path = Path(dir) / f"{p.stem}-{slice_index.item()}.npy"
            np.save(out_path, pred[i].cpu().numpy())

          dist.barrier()

    if self.world_size > 1:
      duration_tensor = torch.tensor([sum_duration, n_duration], device=self.device, dtype=torch.float32)
      dist.all_reduce(duration_tensor, op=dist.ReduceOp.SUM)
      sum_duration, n_duration = duration_tensor[0].item(), int(duration_tensor[1].item())

    return {
      'duration': sum_duration / n_duration,
      **{k: m.compute().item() for k, m in self.metrics.items()}
    }
