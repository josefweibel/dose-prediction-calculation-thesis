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

from metrics import (
	DVHScore,
  DVHAUC,
  DoseScore,
  DoseLevelMetricBase,
  HomogeneityIndex,
  ConformityIndex,
  GradientIndex,
  BodyMaskedMetric,
  ChallengeMAE,
  VarianceOfLaplacian,
  RegionMaskedMetric,
  PTVPrescriptionMAE,
  PTVPrescriptionMeanDose,
  GammaIndex,
  ConstraintSatisfactionRate,
  ConstraintAgreementScore,
  PerOrganConstraintCompliance,
)
from model_sampler import build_sampler
from data import GDPHMM_OAR_V3
from constraints import FractionationType, ConstraintGrade
import numpy as np


def compute_metrics(metrics, preds, target, structure, dataset):
  """
  Update all metrics with predictions and targets.

  This function centralizes the logic for updating different metric types,
  handling special cases like RegionMaskedMetric, BodyMaskedMetric, etc.

  Args:
    metrics: Dictionary of metric name to metric instance
    preds: Predicted doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
    target: True doses, shape (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
    structure: Structure tensor, shape (batch_size, n_channels, H, W) or (batch_size, n_channels, H, W, D)
    dataset: Dataset with selector attribute

  Returns:
    Dictionary mapping metric names to computed values
  """
  results = {}

  for metric_name, metric in metrics.items():
    metric.reset()

    if isinstance(metric, (ConstraintSatisfactionRate, ConstraintAgreementScore, PerOrganConstraintCompliance)):
      # Build organ masks dictionary for constraint metrics
      organ_masks = {}
      for oar_name in GDPHMM_OAR_V3:
        oar_index = dataset.selector.get_oar_index(oar_name)
        organ_masks[oar_name] = structure[:, oar_index]  # (batch_size, H, W) or (batch_size, H, W, D)
      metric.update(preds, target, organ_masks)
    elif isinstance(metric, DoseLevelMetricBase):
      ptv_masks = dataset.selector.get_component(structure, 'ptv_masks')
      ptv70_mask = ptv_masks[:, 0:1]  # (batch_size, 1, H, W) or (batch_size, 1, H, W, D)
      ptv63_mask = ptv_masks[:, 1:2]
      ptv56_mask = ptv_masks[:, 2:3]
      metric.update(preds, target, ptv70_mask, ptv63_mask, ptv56_mask)
    elif isinstance(metric, RegionMaskedMetric):
      metric.update(preds, target, structure)
    elif isinstance(metric, BodyMaskedMetric):
      body = dataset.selector.get_body(structure)
      metric.update(preds, target, body)
    elif isinstance(metric, ChallengeMAE):
      ptv_channel = dataset.selector.get_ptv(structure)
      body = dataset.selector.get_body(structure)
      metric.update(preds, target, ptv_channel[:, 0], body[0])  # Remove channel dim
    else:
      metric.update(preds, target)

    result = metric.compute()

    # Handle metrics that return dictionaries
    if isinstance(result, dict):
      for sub_key, sub_value in result.items():
        full_key = f"{metric_name}/{sub_key}"
        if torch.is_tensor(sub_value):
          results[full_key] = sub_value.item() if not torch.isnan(sub_value) else float('nan')
        else:
          results[full_key] = float(sub_value)
    else:
      results[metric_name] = result.item() if not torch.isnan(result) else float('nan')

  return results


def build_slice_metrics(dataset, device, include_region_metrics=True):
  """Build 2D metrics for slice-by-slice evaluation."""
  dose_range = dataset.dose_range[1] - dataset.dose_range[0]
  dose_min_max = dataset.dose_range  # (min, max) tuple
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
    'dvh_auc': DVHAUC(num_bins=100, normalize=True, dose_range=dose_min_max, sync_on_compute=False),
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

    'ptv70_prescription_mae': PTVPrescriptionMAE(ptv_level=70, sync_on_compute=False),
    'ptv63_prescription_mae': PTVPrescriptionMAE(ptv_level=63, sync_on_compute=False),
    'ptv56_prescription_mae': PTVPrescriptionMAE(ptv_level=56, sync_on_compute=False),
    'ptv70_mean_dose': PTVPrescriptionMeanDose(ptv_level=70, sync_on_compute=False),
    'ptv63_mean_dose': PTVPrescriptionMeanDose(ptv_level=63, sync_on_compute=False),
    'ptv56_mean_dose': PTVPrescriptionMeanDose(ptv_level=56, sync_on_compute=False),

    'ptv70_conformity_index': ConformityIndex(ptv_level=70, sync_on_compute=False),
    'ptv63_conformity_index': ConformityIndex(ptv_level=63, sync_on_compute=False),
    'ptv56_conformity_index': ConformityIndex(ptv_level=56, sync_on_compute=False),

    'ptv70_gradient_index': GradientIndex(ptv_level=70, sync_on_compute=False),
    'ptv63_gradient_index': GradientIndex(ptv_level=63, sync_on_compute=False),
    'ptv56_gradient_index': GradientIndex(ptv_level=56, sync_on_compute=False),
  }

  if include_region_metrics:
    # Build region list with indices for RegionMaskedMetric
    ptv_start, _ = dataset.selector.get_channel_indices('ptv')
    regions = [('PTV', ptv_start)]

    for oar_name in GDPHMM_OAR_V3:
      oar_index = dataset.selector.get_oar_index(oar_name)
      regions.append((oar_name, oar_index))

    region_metric_configs = [
      ('mse', lambda: MeanSquaredError(sync_on_compute=False)),
      ('rmse', lambda: MeanSquaredError(squared=False, sync_on_compute=False)),
      ('psnr', lambda: PeakSignalNoiseRatio(data_range=dose_range, sync_on_compute=False)),
      ('dvh', lambda: DVHScore(sync_on_compute=False)),
      ('dvh_auc', lambda: DVHAUC(num_bins=100, normalize=True, dose_range=dose_min_max, sync_on_compute=False)),
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

  # Add constraint-based metrics
  metrics['csr_all'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['csr_mandatory'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    grade_filter=ConstraintGrade.MANDATORY,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['csr_recommended'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    grade_filter=ConstraintGrade.RECOMMENDED,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['csr_optimal'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    grade_filter=ConstraintGrade.OPTIMAL,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['csr_acceptable'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    grade_filter=ConstraintGrade.ACCEPTABLE,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['agreement'] = ConstraintAgreementScore(
    organ_names=GDPHMM_OAR_V3,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )

  # Add per-region PerOrganConstraintCompliance metrics
  if include_region_metrics:
    for region_name, region_index in regions:
      if region_name == 'PTV':
        continue

      region_compliance = PerOrganConstraintCompliance(
        organ_name=region_name,
        fractionation=FractionationType.CONVENTIONAL,
        sync_on_compute=False
      )
      metrics[f'compliance/{region_name}'] = region_compliance

  return {k: v.to(device) for k, v in metrics.items()}


def build_volume_metrics(dataset, device, include_region_metrics=True):
  """Build 3D volume metrics for whole-volume evaluation."""
  dose_min_max = dataset.dose_range  # (min, max) tuple

  # Volume-based metrics (no SSIM/MS-SSIM/PSNR as they are 2D metrics)
  metrics = {
    'vol_mse': MeanSquaredError(sync_on_compute=False),
    'vol_rmse': MeanSquaredError(squared=False, sync_on_compute=False),
    'vol_dvh': DVHScore(sync_on_compute=False),
    'vol_dvh_auc': DVHAUC(num_bins=100, normalize=True, dose_range=dose_min_max, sync_on_compute=False),
    'vol_dose_score': DoseScore(sync_on_compute=False),
    'vol_homogeneity_index': HomogeneityIndex(sync_on_compute=False),

    'vol_mse_body': BodyMaskedMetric(MeanSquaredError(sync_on_compute=False), sync_on_compute=False),
    'vol_rmse_body': BodyMaskedMetric(MeanSquaredError(squared=False, sync_on_compute=False), sync_on_compute=False),
    'vol_dose_score_body': BodyMaskedMetric(DoseScore(sync_on_compute=False), sync_on_compute=False),

    'vol_mae_challenge': ChallengeMAE(sync_on_compute=False),

    'vol_ptv70_prescription_mae': PTVPrescriptionMAE(ptv_level=70, sync_on_compute=False),
    'vol_ptv63_prescription_mae': PTVPrescriptionMAE(ptv_level=63, sync_on_compute=False),
    'vol_ptv56_prescription_mae': PTVPrescriptionMAE(ptv_level=56, sync_on_compute=False),
    'vol_ptv70_mean_dose': PTVPrescriptionMeanDose(ptv_level=70, sync_on_compute=False),
    'vol_ptv63_mean_dose': PTVPrescriptionMeanDose(ptv_level=63, sync_on_compute=False),
    'vol_ptv56_mean_dose': PTVPrescriptionMeanDose(ptv_level=56, sync_on_compute=False),

    'vol_ptv70_conformity_index': ConformityIndex(ptv_level=70, sync_on_compute=False),
    'vol_ptv63_conformity_index': ConformityIndex(ptv_level=63, sync_on_compute=False),
    'vol_ptv56_conformity_index': ConformityIndex(ptv_level=56, sync_on_compute=False),

    'vol_ptv70_gradient_index': GradientIndex(ptv_level=70, sync_on_compute=False),
    'vol_ptv63_gradient_index': GradientIndex(ptv_level=63, sync_on_compute=False),
    'vol_ptv56_gradient_index': GradientIndex(ptv_level=56, sync_on_compute=False),

    'vol_gamma_5_5': GammaIndex(dose_tolerance=5.0, distance_tolerance=5.0, sync_on_compute=False),
    'vol_gamma_3_3': GammaIndex(dose_tolerance=3.0, distance_tolerance=3.0, sync_on_compute=False),
    'vol_gamma_2_2': GammaIndex(dose_tolerance=2.0, distance_tolerance=2.0, sync_on_compute=False),
    'vol_gamma_1_1': GammaIndex(dose_tolerance=1.0, distance_tolerance=1.0, sync_on_compute=False),
  }

  if include_region_metrics:
    # Build region list with indices for RegionMaskedMetric
    ptv_start, _ = dataset.selector.get_channel_indices('ptv')
    regions = [('PTV', ptv_start)]

    for oar_name in GDPHMM_OAR_V3:
      oar_index = dataset.selector.get_oar_index(oar_name)
      regions.append((oar_name, oar_index))

    region_metric_configs = [
      ('mse', lambda: MeanSquaredError(sync_on_compute=False)),
      ('rmse', lambda: MeanSquaredError(squared=False, sync_on_compute=False)),
      ('dvh', lambda: DVHScore(sync_on_compute=False)),
      ('dvh_auc', lambda: DVHAUC(num_bins=100, normalize=True, dose_range=dose_min_max, sync_on_compute=False)),
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
        metrics[f'vol_{metric_name}/{region_name}'] = region_metric

  # Add constraint-based metrics
  metrics['vol_csr_all'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['vol_csr_mandatory'] = ConstraintSatisfactionRate(
    organ_names=GDPHMM_OAR_V3,
    grade_filter=ConstraintGrade.MANDATORY,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )
  metrics['vol_agreement'] = ConstraintAgreementScore(
    organ_names=GDPHMM_OAR_V3,
    fractionation=FractionationType.CONVENTIONAL,
    sync_on_compute=False
  )

  if include_region_metrics:
    for region_name, region_index in regions:
      if region_name == 'PTV':
        continue

      region_compliance = PerOrganConstraintCompliance(
        organ_name=region_name,
        fractionation=FractionationType.CONVENTIONAL,
        sync_on_compute=False
      )
      metrics[f'vol_compliance/{region_name}'] = region_compliance

  return {k: v.to(device) for k, v in metrics.items()}


def compute_volume_metrics(samples_path, dataset, device, rank=0, world_size=1):
  """
  Compute volume metrics for all combined volume files in a samples directory.

  Args:
    samples_path: Path to the samples directory containing combined .npy volume files
    dataset: Dataset instance for accessing structure and target data
    device: Device to run computations on
    rank: Process rank for distributed computing
    world_size: Total number of processes for distributed computing

  Returns:
    DataFrame with volume metrics for each sample
  """
  samples_path = Path(samples_path)
  volume_files = sorted(list(samples_path.glob('*.npy')))
  # Exclude the _sample_metrics.csv and _overall_metrics.csv indicator files
  volume_files = [f for f in volume_files if not f.stem.startswith('_')]

  if len(volume_files) == 0:
    if rank == 0:
      print(f'No volume files found in {samples_path}')
    return pd.DataFrame()

  # Distribute volume files across ranks
  # Each rank processes a subset of the volumes
  rank_volume_files = volume_files[rank::world_size]

  if rank == 0:
    print(f'Computing volume metrics for {len(volume_files)} volumes across {world_size} GPUs...')
  print(f'Rank {rank} processing {len(rank_volume_files)} volumes')

  metrics = build_volume_metrics(dataset, device)
  volume_metrics = []

  for volume_file in tqdm(rank_volume_files, desc=f'Computing volume metrics (Rank {rank})', disable=rank != 0):
    # Load predicted volume: shape (1, H, W, D) where D is num_slices
    pred_volume = np.load(volume_file)
    pred_volume = torch.from_numpy(pred_volume).to(device)

    # Get sample name from filename
    sample_name = volume_file.stem

    # Find all slices for this sample in the dataset
    sample_rows = dataset.index[dataset.index['volume'] == sample_name]

    if len(sample_rows) == 0:
      print(f'Warning: No slices found for {sample_name}')
      continue

    # Sort by slice index to ensure correct ordering
    sample_rows = sample_rows.sort_values('slice_idx')
    sample_indices = sample_rows.index.tolist()

    # Load all slices to reconstruct structure and target volumes
    structure_slices = []
    target_slices = []

    for idx in sample_indices:
      structure, target, _ = dataset[idx]
      structure_slices.append(structure)
      target_slices.append(target)

    # Stack slices along depth dimension: (C, H, W, D)
    structure_volume = torch.stack(structure_slices, dim=-1).to(device)
    target_volume = torch.stack(target_slices, dim=-1).to(device)

    # Denormalize target
    target_volume = dataset.denormalize_target(target_volume)

    # Add batch dimension for metrics computation: (1, C, H, W, D)
    structure_volume = structure_volume.unsqueeze(0)
    target_volume = target_volume.unsqueeze(0)
    pred_volume = pred_volume.unsqueeze(0)

    metric_results = compute_metrics(
      metrics,
      pred_volume,
      target_volume,
      structure_volume,
      dataset
    )

    volume_metrics.append({
      'file_name': sample_name,
      **metric_results
    })

  # Gather results from all ranks if in distributed mode
  if world_size > 1:
    # Ensure all ranks finish processing before gathering
    dist.barrier()

    all_volume_metrics = [None] * world_size
    dist.all_gather_object(all_volume_metrics, volume_metrics)

    if rank == 0:
      # Flatten the list of lists
      volume_metrics = [item for sublist in all_volume_metrics for item in sublist]

  return pd.DataFrame(volume_metrics)


class Evaluator:
  def __init__(self, dataloader, model_config, sampling_config, device, rank=0, world_size=1):
    self.dataloader = dataloader
    self.model_config = model_config
    self.sampling_config = sampling_config
    self.device = device
    self.rank = rank
    self.world_size = world_size
    self.conditioning_groups = self.model_config.get('conditioning', {}).get('types', ['ct', 'ptv', 'oar'])

    self.metrics = build_slice_metrics(dataloader.dataset, device)

  def evaluate(self, model, dir=None):
    model.eval()
    sampler = build_sampler(self.model_config, self.sampling_config, model).to(self.device)

    with torch.no_grad():
      sample_metrics = []

      pbar = tqdm(self.dataloader, desc='Evaluating', leave=False, disable=self.rank != 0)
      for structure, target, info in pbar:
        structure = structure.to(self.device)
        target = target.to(self.device)
        target = self.dataloader.dataset.denormalize_target(target)

        body = self.dataloader.dataset.selector.get_body(structure)
        condition = self.dataloader.dataset.selector.select_channels(structure, self.conditioning_groups)

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
          sample_body = body[i:i+1]

          sample_pred = sample_pred * sample_body

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
      overall_metrics.to_frame().to_csv(dir / '_overall_metrics.csv')

    dist.barrier()

    return sample_metrics, overall_metrics
