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

from metrics import DVHScore, DoseScore, HomogeneityIndex, BodyMaskedMetric, ChallengeMAE, VarianceOfLaplacian, RegionMaskedMetric
from model_sampler import build_sampler
from data import GDPHMM_OAR


def compute_metrics(metrics, preds, target, structure, dataset):
  """
  Update all metrics with predictions and targets.

  This function centralizes the logic for updating different metric types,
  handling special cases like RegionMaskedMetric, BodyMaskedMetric, etc.

  Args:
    metrics: Dictionary of metric name to metric instance
    preds: Predicted doses, shape (batch_size, 1, H, W)
    target: True doses, shape (batch_size, 1, H, W)
    structure: Structure tensor, shape (batch_size, n_channels, H, W)
    dataset: Dataset with selector attribute

  Returns:
    Dictionary mapping metric names to computed values
  """
  results = {}

  for metric_name, metric in metrics.items():
    metric.reset()

    if isinstance(metric, RegionMaskedMetric):
      metric.update(preds, target, structure)
    elif isinstance(metric, BodyMaskedMetric):
      body = dataset.selector.get_body(structure)
      metric.update(preds, target, body)
    elif isinstance(metric, HomogeneityIndex):
      ptv_channel = dataset.selector.get_ptv(structure)
      metric.update(preds, target, ptv_channel[:, 0])  # Remove channel dim
    elif isinstance(metric, ChallengeMAE):
      ptv_channel = dataset.selector.get_ptv(structure)
      body = dataset.selector.get_body(structure)
      metric.update(preds, target, ptv_channel[:, 0], body[0])  # Remove channel dim
    else:
      metric.update(preds, target)

    result = metric.compute()
    # Handle NaN values (when region is not present in slice)
    results[metric_name] = result.item() if not torch.isnan(result) else float('nan')

  return results


def build_metrics(dataset, device, include_region_metrics=True):
  dose_range = dataset.dose_range[1] - dataset.dose_range[0]
  ms_ssim_kwargs = dict(
    data_range=dose_range,
    betas=(0.0448, 0.2856, 0.3001, 0.2363), # only 4 values since image size is 160x176
  )

  metrics = {
    'mse': MeanSquaredError(sync_on_compute=False),
    'rmse': MeanSquaredError(squared=False, sync_on_compute=False),
    'psnr': PeakSignalNoiseRatio(data_range=dose_range, sync_on_compute=False),
    'ssim': StructuralSimilarityIndexMeasure(data_range=dose_range, sync_on_compute=False),
    'ms_ssim': MultiScaleStructuralSimilarityIndexMeasure(**ms_ssim_kwargs, sync_on_compute=False),
    'dvh': DVHScore(sync_on_compute=False),
    'dose_score': DoseScore(sync_on_compute=False),
    'homogeneity_index': HomogeneityIndex(sync_on_compute=False),

    'mse_body': BodyMaskedMetric(MeanSquaredError(sync_on_compute=False), sync_on_compute=False),
    'rmse_body': BodyMaskedMetric(MeanSquaredError(squared=False, sync_on_compute=False), sync_on_compute=False),
    'psnr_body': BodyMaskedMetric(PeakSignalNoiseRatio(data_range=dose_range, sync_on_compute=False), sync_on_compute=False),
    'dose_score_body': BodyMaskedMetric(DoseScore(sync_on_compute=False), sync_on_compute=False),

    'mae_challenge': ChallengeMAE(sync_on_compute=False),

    'var_laplacian_pred': VarianceOfLaplacian(mode='preds', sync_on_compute=False),
    'var_laplacian_target': VarianceOfLaplacian(mode='target', sync_on_compute=False),
    'var_laplacian_diff': VarianceOfLaplacian(mode='diff', sync_on_compute=False),
  }

  if include_region_metrics:
    # Build region list with indices for RegionMaskedMetric
    ptv_start, _ = dataset.selector.get_channel_indices('ptv')
    regions = [('PTV', ptv_start)]

    for oar_name in GDPHMM_OAR:
      oar_index = dataset.selector.get_oar_index(oar_name)
      regions.append((oar_name, oar_index))

    region_metric_configs = [
      ('mse', lambda: MeanSquaredError(sync_on_compute=False)),
      ('rmse', lambda: MeanSquaredError(squared=False, sync_on_compute=False)),
      ('psnr', lambda: PeakSignalNoiseRatio(data_range=dose_range, sync_on_compute=False)),
      ('dvh', lambda: DVHScore(sync_on_compute=False)),
      ('dose_score', lambda: DoseScore(sync_on_compute=False)),
    ]

    for region_name, region_index in regions:
      for metric_name, metric_factory in region_metric_configs:
        region_metric = RegionMaskedMetric(
          base_metric=metric_factory(),
          region_name=region_name,
          region_index=region_index,
          sync_on_compute=False
        )
        metrics[f'{metric_name}/{region_name}'] = region_metric

  return {k: v.to(device) for k, v in metrics.items()}


class Evaluator:
  def __init__(self, dataloader, model_config, sampling_config, device, rank=0, world_size=1):
    self.dataloader = dataloader
    self.model_config = model_config
    self.sampling_config = sampling_config
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.conditioning_groups = self.model_config.get('conditioning', {}).get('types', ['ct', 'ptv', 'oar'])

    self.metrics = build_metrics(dataloader.dataset, device)

  def evaluate(self, model, dir=None):
    model.eval()
    sampler = build_sampler(self.model_config, self.sampling_config, model).to(self.device)

    with torch.no_grad():
      sample_metrics = []

      pbar = tqdm(self.dataloader, desc='Evaluating', leave=False, disable=self.rank != 0)
      for batch_idx, (structure, target, info) in enumerate(pbar):
        structure = structure.to(self.device)
        target = target.to(self.device)
        target = self.dataloader.dataset.denormalize_target(target)
        body = self.dataloader.dataset.selector.get_body(structure)

        condition = self.dataloader.dataset.selector.select_channels(structure, self.conditioning_groups)

        # Debug: Print shapes on first batch
        if batch_idx == 0 and self.rank == 0:
          print(f"\n[DEBUG] Evaluation batch 0:")
          print(f"  structure.shape: {structure.shape}")
          print(f"  condition.shape: {condition.shape}")
          print(f"  conditioning_groups: {self.conditioning_groups}")
          print(f"  validation dataset groups: {self.dataloader.dataset.selector.groups}")
          print(f"  validation dataset total_channels: {self.dataloader.dataset.selector.total_channels}")

          # Check condition data statistics
          print(f"\n[DEBUG] Condition statistics (first sample in batch):")
          print(f"  CT channel - min: {condition[0, 0].min().item():.4f}, max: {condition[0, 0].max().item():.4f}, mean: {condition[0, 0].mean().item():.4f}")
          print(f"  PTV_SDM channel - min: {condition[0, 1].min().item():.4f}, max: {condition[0, 1].max().item():.4f}, mean: {condition[0, 1].mean().item():.4f}")
          print(f"  OAR_SDM first channel - min: {condition[0, 2].min().item():.4f}, max: {condition[0, 2].max().item():.4f}, mean: {condition[0, 2].mean().item():.4f}")

        autocast = contextlib.nullcontext()
        if self.sampling_config['precision'] == 'fp16_amp':
          autocast = torch.amp.autocast('cuda', dtype=torch.float16)
        elif self.sampling_config['precision'] == 'bf16_amp':
          autocast = torch.amp.autocast('cuda', dtype=torch.bfloat16)

        with autocast:
          start = time()
          x_T = torch.randn_like(target, device=self.device)
          pred = sampler(x_T, condition)
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

          # Update all metrics and get results
          metric_results = compute_metrics(
            self.metrics,
            sample_pred,
            sample_target,
            structure[i:i+1],
            self.dataloader.dataset
          )
          sample_row.update(metric_results)

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
