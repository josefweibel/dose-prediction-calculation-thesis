from time import time
import torch
import torch.distributed as dist
from torchmetrics import MeanSquaredError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, MultiScaleStructuralSimilarityIndexMeasure
from tqdm import tqdm
import numpy as np
from pathlib import Path
import pandas as pd
import contextlib

from metrics import DVHScore, DoseScore, HomogeneityIndex
from model_sampler import build_sampler

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
      'mse': MeanSquaredError(sync_on_compute=False),
      'rmse': MeanSquaredError(squared=False, sync_on_compute=False),
      'psnr': PeakSignalNoiseRatio(data_range=dose_range, sync_on_compute=False),
      'ssim': StructuralSimilarityIndexMeasure(data_range=dose_range, sync_on_compute=False),
      'ms_ssim': MultiScaleStructuralSimilarityIndexMeasure(
        data_range=dataloader.dataset.dose_range,
        betas=(0.0448, 0.2856, 0.3001, 0.2363), # only 4 values since image size is 160x176
        sync_on_compute=False,
      ),
      'dvh': DVHScore(sync_on_compute=False),
      'dose_score': DoseScore(sync_on_compute=False),
      'homogeneity_index': HomogeneityIndex(sync_on_compute=False)
    }

    self.metrics = {k: v.to(device) for k, v in self.metrics.items()}

  def evaluate(self, model, dir=None):
    model.eval()
    sampler = build_sampler(self.model_config, self.sampling_config, model).to(self.device)

    with torch.no_grad():
      sample_metrics = []

      pbar = tqdm(self.dataloader, desc='Evaluating', leave=False, disable=self.rank != 0)
      for structure, target, body, info in pbar:
        structure = structure.to(self.device)
        target = target.to(self.device)
        target = self.dataloader.dataset.denormalize_target(target)
        body = body.to(self.device)

        autocast = contextlib.nullcontext()
        if self.sampling_config['precision'] == 'fp16_amp':
          autocast = torch.amp.autocast('cuda', dtype=torch.float16)
        elif self.sampling_config['precision'] == 'bf16_amp':
          autocast = torch.amp.autocast('cuda', dtype=torch.bfloat16)

        with autocast:
          start = time()
          x_T = torch.randn_like(target, device=self.device)
          pred = sampler(x_T, structure)
          duration = time() - start

        pred = torch.clamp(pred, 0., 1.)
        pred = self.dataloader.dataset.denormalize_target(pred)

        # Calculate metrics for each sample in the batch
        for i in range(target.shape[0]):
          sample_pred = pred[i:i+1]  # Keep batch dimension
          sample_target = target[i:i+1]

          sample_row = {
            'file_name': info['file_name'][i],
            'slice_index': info['index'][i].item(),
            'duration': duration / target.shape[0]
          }

          for metric_name, metric in self.metrics.items():
            metric.reset()

            m_kwargs = {'mask': structure[i:i+1, 1]} if metric_name == 'homogeneity_index' else {}
            metric.update(sample_pred, sample_target, **m_kwargs)

            sample_row[metric_name] = metric.compute().item()

          sample_metrics.append(sample_row)

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
      all_sample_metrics = [None] * self.world_size
      dist.all_gather_object(all_sample_metrics, sample_metrics)

      if self.rank == 0:
        sample_metrics = [item for sublist in all_sample_metrics for item in sublist]

    dist.barrier()

    sample_metrics = pd.DataFrame(sample_metrics).drop_duplicates(['file_name', 'slice_index'])
    overall_metrics = sample_metrics.mean(numeric_only=True)

    if dir is not None and self.rank == 0:
      sample_metrics.to_csv(dir / '_sample_metrics.csv', index=False)
      overall_metrics.to_frame().T.to_csv(dir / '_overall_metrics.csv', index=False)

    dist.barrier()

    return sample_metrics, overall_metrics
