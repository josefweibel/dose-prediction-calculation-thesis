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
  ):
    self.model_config = read_model_config(model_name)
    self.sampling_config = read_sampling_config(sampling_name)
    self.data_split = data_split
    self.epoch = epoch
    self.rank = rank if rank is not None else 0
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    self.dataset = create_gdphmm_dataset(self.model_config, data_split)

    self.device = detect_device(local_rank)
    self.metrics = build_slice_metrics(self.dataset, self.device)

    self.samples_path = SAMPLES_PATH / model_name / sampling_name / f'{data_split}_{epoch}'
    self.old_scores = pd.read_csv(self.samples_path / f'_sample_metrics.csv')

  def calculate(self):
    # Recalculate 2D slice metrics
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

      sample_metrics = pd.DataFrame(sample_metrics).drop_duplicates(['file_name', 'slice_index'])
      overall_metrics = sample_metrics.mean(numeric_only=True)

      sample_metrics.to_csv(self.samples_path / '_sample_metrics.csv', index=False)
      overall_metrics.to_frame().to_csv(self.samples_path / '_overall_metrics.csv')

    if self.is_distributed:
      dist.barrier()

    # Recalculate volume metrics
    print(f'\nRank {self.rank}: Recalculating volume metrics...')
    volume_metrics = compute_volume_metrics(
      self.samples_path,
      self.dataset,
      self.device,
      rank=self.rank,
      world_size=self.world_size
    )

    if self.is_distributed:
      dist.barrier()

    if self.is_main_process:
      if len(volume_metrics) > 0:
        # Compute overall volume metrics
        overall_volume_metrics = volume_metrics.mean(numeric_only=True)

        # Backup old volume metrics if they exist
        if (self.samples_path / '_volume_metrics.csv').exists():
          (self.samples_path / '_volume_metrics.csv').rename(self.samples_path / '_volume_metrics_old.csv')
        if (self.samples_path / '_overall_volume_metrics.csv').exists():
          (self.samples_path / '_overall_volume_metrics.csv').rename(self.samples_path / '_overall_volume_metrics_old.csv')

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


