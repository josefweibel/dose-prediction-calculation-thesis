import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from evaluator import build_metrics, compute_metrics
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
  ):
    self.model_config = read_model_config(model_name)
    self.sampling_config = read_sampling_config(sampling_name)
    self.data_split = data_split
    self.epoch = epoch

    self.dataset = create_gdphmm_dataset(self.model_config, data_split)

    self.device = detect_device(None)
    self.metrics = build_metrics(self.dataset, self.device)

    self.samples_path = SAMPLES_PATH / model_name / sampling_name / f'{data_split}_{epoch}'
    self.old_scores = pd.read_csv(self.samples_path / f'_sample_metrics.csv')

  def calculate(self):
    sample_metrics = []

    for idx in tqdm(range(len(self.dataset))):
      structure, target, info = self.dataset[idx]
      structure = structure.unsqueeze(0).to(self.device)
      target = target.unsqueeze(0).to(self.device)
      target = self.dataset.denormalize_target(target)
      body = self.dataset.selector.get_body(structure).to(self.device)

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
          body
        )
        sample_row.update(metric_results)

      sample_metrics.append(sample_row)

    (self.samples_path / '_sample_metrics.csv').rename(self.samples_path / '_sample_metrics_old.csv')
    (self.samples_path / '_overall_metrics.csv').rename(self.samples_path / '_overall_metrics_old.csv')

    sample_metrics = pd.DataFrame(sample_metrics).drop_duplicates(['file_name', 'slice_index'])
    overall_metrics = sample_metrics.mean(numeric_only=True)

    sample_metrics.to_csv(self.samples_path / '_sample_metrics.csv', index=False)
    overall_metrics.to_frame().T.to_csv(self.samples_path / '_overall_metrics.csv', index=False)

  def finish(self):
    print('finishing metrics recalculator instance', flush=True)

    self.dataset.finish()

    torch.cuda.empty_cache()


