import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from tqdm import tqdm

from evaluator import build_slice_metrics, compute_metrics, compute_volume_metrics
from results import SAMPLES_PATH
from utils.mp import detect_device
from config import read_model_config, read_sampling_config
from data import create_gdphmm_dataset


class MetricsRecalculator:
  def __init__(
    self,
    model_name,
    sampling_name,
    data_split,
    epoch,
    rank=None,
    world_size=None,
    local_rank=None,
    metric_names=None,
    skip_slice_metrics=False,
    skip_volume_metrics=False,
  ):
    self.model_config = read_model_config(model_name)
    self.sampling_config = read_sampling_config(sampling_name)
    self.data_split = data_split
    self.epoch = epoch
    self.rank = rank if rank is not None else 0
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0
    self.metric_names = metric_names
    self.skip_slice_metrics = skip_slice_metrics
    self.skip_volume_metrics = skip_volume_metrics

    self.dataset = create_gdphmm_dataset(self.model_config, data_split)

    self.device = detect_device()

    if not skip_slice_metrics:
      self.metrics = build_slice_metrics(self.dataset, self.device, metric_names=metric_names)

    self.samples_path = SAMPLES_PATH / model_name / sampling_name / f'{data_split}_{epoch}'
    self.old_scores = pd.read_csv(self.samples_path / f'_sample_metrics.csv')

  def _merge_metrics(self, old_df, new_df, key_cols):
    """Merge new metric columns into old DataFrame, keeping non-recomputed columns."""
    metric_cols = [c for c in new_df.columns if c not in key_cols]
    merged = old_df.set_index(key_cols).copy()
    new_indexed = new_df.set_index(key_cols)

    for col in metric_cols:
      merged[col] = new_indexed[col]

    return merged.reset_index()

  def calculate(self):
    if not self.skip_slice_metrics:
      self._calculate_slice_metrics()

    if self.is_distributed:
      dist.barrier()

    if not self.skip_volume_metrics:
      self._calculate_volume_metrics()

  def _calculate_slice_metrics(self):
    if self.is_main_process:
      print('\nRecalculating 2D slice metrics...')

    sample_metrics = []

    # Distribute dataset indices across ranks
    dataset_size = len(self.dataset)
    rank_indices = list(range(self.rank, dataset_size, self.world_size))

    if self.is_main_process:
      print(f'Recalculating 2D metrics for {dataset_size} slices across {self.world_size} GPUs...')
    print(f'Rank {self.rank} processing {len(rank_indices)} slices')

    for idx in tqdm(rank_indices, desc=f'Recalculating 2D metrics (Rank {self.rank})', disable=self.rank != 0):
      structure, target, info = self.dataset[idx]
      structure = structure.unsqueeze(0).to(self.device)
      target = target.unsqueeze(0).to(self.device)
      target = self.dataset.denormalize_target(target)

      file_name = info['file_name']
      slice_idx = info['index']

      sample_row = {
        'file_name': file_name,
        'slice_index': slice_idx,
        'duration': self.old_scores.loc[(self.old_scores['file_name'] == file_name) & (self.old_scores['slice_index'] == slice_idx), 'duration'].values[0],
      }

      with torch.no_grad():
        pred = np.load(self.samples_path / file_name.replace(".npz", ".npy"))[..., slice_idx]
        pred = torch.Tensor(pred).unsqueeze(0).to(self.device)

        metric_results = compute_metrics(
          self.metrics,
          pred,
          target,
          structure,
          self.dataset
        )
        sample_row.update(metric_results)

      sample_metrics.append(sample_row)

    # Gather results from all ranks if in distributed mode
    if self.world_size > 1:
      # Ensure all ranks finish processing before gathering
      dist.barrier()

      all_sample_metrics = [None] * self.world_size
      dist.all_gather_object(all_sample_metrics, sample_metrics)

      if self.rank == 0:
        # Flatten the list of lists
        sample_metrics = [item for sublist in all_sample_metrics for item in sublist]

    if self.is_distributed:
      dist.barrier()

    if self.is_main_process:
      # Backup old metrics
      (self.samples_path / '_sample_metrics.csv').rename(self.samples_path / '_sample_metrics_old.csv')
      (self.samples_path / '_overall_metrics.csv').rename(self.samples_path / '_overall_metrics_old.csv')

      new_df = pd.DataFrame(sample_metrics).drop_duplicates(['file_name', 'slice_index'])

      if self.metric_names is not None:
        sample_metrics = self._merge_metrics(self.old_scores, new_df, ['file_name', 'slice_index'])
      else:
        sample_metrics = new_df

      overall_metrics = sample_metrics.mean(numeric_only=True)

      sample_metrics.to_csv(self.samples_path / '_sample_metrics.csv', index=False)
      overall_metrics.to_frame().to_csv(self.samples_path / '_overall_metrics.csv')

    if self.is_distributed:
      dist.barrier()

  def _calculate_volume_metrics(self):
    print(f'\nRank {self.rank}: Recalculating volume metrics...')
    volume_metrics = compute_volume_metrics(
      self.samples_path,
      self.dataset,
      self.device,
      rank=self.rank,
      world_size=self.world_size,
      metric_names=self.metric_names,
    )

    if self.is_distributed:
      dist.barrier()

    if self.is_main_process:
      if len(volume_metrics) > 0:
        old_volume_path = self.samples_path / '_volume_metrics.csv'
        old_overall_volume_path = self.samples_path / '_overall_volume_metrics.csv'

        if self.metric_names is not None and old_volume_path.exists():
          old_volume_df = pd.read_csv(old_volume_path)
          volume_metrics = self._merge_metrics(old_volume_df, volume_metrics, ['file_name'])

        overall_volume_metrics = volume_metrics.mean(numeric_only=True)

        # Backup old volume metrics if they exist
        if old_volume_path.exists():
          old_volume_path.rename(self.samples_path / '_volume_metrics_old.csv')
        if old_overall_volume_path.exists():
          old_overall_volume_path.rename(self.samples_path / '_overall_volume_metrics_old.csv')

        # Save volume metrics
        volume_metrics.to_csv(self.samples_path / '_volume_metrics.csv', index=False)
        overall_volume_metrics.to_frame().to_csv(self.samples_path / '_overall_volume_metrics.csv')

        print(f'Volume metrics saved to {self.samples_path}')
      else:
        print('No volume metrics computed')

    if self.is_distributed:
      dist.barrier()

  def finish(self):
    print('finishing metrics recalculator instance', flush=True)

    self.dataset.finish()

    torch.cuda.empty_cache()
