import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from itertools import cycle
from multiprocessing import Manager
from pathlib import Path

from data import GDPHMMSlicesDataset, SharedCache, GDPHMM_OAR


SAMPLES_PATH = Path(__file__).parents[1] / 'samples'

METRC_LABELS = {
  'mse': 'Mean Squared Error (MSE)',
  'rmse': 'Root Mean Squared Error (RMSE)',
  'psnr': 'Peak Signal-to-Noise Ratio (PSNR)',
  'ssim': 'Structural Similarity Index Measure (SSIM)',
  'ms_ssim': 'Multi-Scale Structural Similarity Index Measure (MS-SSIM)',
  'dvh': 'Dose-Volume Histogram Score (DVH Score)',
  'dose_score': 'Dose Score (MAE)',
  'homogeneity_index': 'Homogeneity Index',
}

METRIC_ABBRS = {
  'mse': 'MSE',
  'rmse': 'RMSE',
  'psnr': 'PSNR',
  'ssim': 'SSIM',
  'ms_ssim': 'MS-SSIM',
  'dvh': 'DVH',
  'dose_score': 'Dose Score (MAE)',
  'homogeneity_index': 'HI',
}

DEFAULT_METRICS = ['rmse', 'dose_score', 'dvh', 'homogeneity_index', 'ssim', 'psnr']


def compute_dvh(dose, mask, bins):
  """Compute differential DVH (volume fraction ≥ dose)."""
  masked_dose = dose[mask > 0] if mask is not None else dose

  if masked_dose.size == 0:
    return bins[:-1], np.zeros_like(bins[:-1])

  hist, bin_edges = np.histogram(masked_dose, bins=bins)
  cdf = np.cumsum(hist[::-1])[::-1] / np.sum(hist)
  return bin_edges[:-1], cdf

manager = Manager()

class BaseResults:
  def __init__(self, model_name, sampling_name, epoch, dataset_split):
    self.model_name = model_name
    self.sampling_name = sampling_name
    self.epoch = epoch
    self.dataset_split = dataset_split

    self.base_path = SAMPLES_PATH / model_name / sampling_name / f'{dataset_split}_{epoch}'

    cache = SharedCache(manager)
    self.dataset = GDPHMMSlicesDataset(
      dataset_name='gdphmm-hn-vmat',
      split=dataset_split,
      img_size=[176, 160],
      cache=cache,
    )

  def plot_quality_over_slice_index(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    grouped = self.slice_metrics.groupby('slice_index')[metric]
    means = grouped.mean()
    stds = grouped.std()

    ax.errorbar(means.index, means.values, yerr=stds.values, fmt='o', ecolor='red', capsize=3)
    ax.set_title(f'{METRIC_ABBRS.get(metric, metric)} vs Slice Index (Mean ± SD) for Model {self.model_name}')
    ax.set_xlabel('Slice Index')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))

    return ax

  def plot_organ_dvh(self, organ_name, ax=None, max_dose=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    structure_index = GDPHMM_OAR.index(organ_name) + 2  # +2 to skip CT and PTV

    original_doses = []
    pred_doses = []

    for file_name in self.slice_metrics['file_name'].unique():
      sample_results = self.for_sample(file_name.replace('.npz', ''))
      structure, original_dose = sample_results.load_orig_volume()
      pred_dose = sample_results.load_pred_volume()

      eye_mask = structure[structure_index]
      original_dose = original_dose[eye_mask > 0]
      pred_dose = pred_dose[eye_mask > 0]

      original_doses.extend(original_dose)
      pred_doses.extend(pred_dose)

    if len(original_doses) == 0 or len(pred_doses) == 0:
      print(f'No data found for organ: {organ_name}')
      return ax

    bins = np.linspace(0, max(*original_doses, *pred_doses, max_dose or 0), 200)
    x, y_orig = compute_dvh(np.array(original_doses), None, bins)
    _, y_pred = compute_dvh(np.array(pred_doses), None, bins)
    ax.plot(x, y_orig, color='blue', linestyle='-', label='Original Dose')
    ax.plot(x, y_pred, color='green', linestyle='-', label='Predicted Dose')

    # TODO show max mean of all samples
    ax.axvline(x=np.mean(original_doses), color='blue', linestyle='--', label='Mean Original Dose')
    ax.axvline(x=np.mean(pred_doses), color='green', linestyle='--', label='Mean Predicted Dose')

    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume Fraction ≥ Dose')
    ax.set_title(f'{organ_name} DVH Comparison')
    ax.grid(True, linestyle=':')
    ax.legend()

    return ax

  def plot_eye_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Eyes', ax, max_dose=45)

    ax.axvline(x=45, color='orange', linestyle='-', label=r'$D_{MAX} < 45$ Gy Constraint')
    ax.legend()

    return ax

  def plot_cochlea_l_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Cochlea_L', ax, max_dose=45)

    ax.axvline(x=45, color='orange', linestyle='--', label=r'$D_{MEAN} < 45$ Gy Constraint')
    ax.legend()

    return ax

  def plot_cochlea_r_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Cochlea_R', ax, max_dose=45)

    ax.axvline(x=45, color='orange', linestyle='--', label=r'$D_{MEAN} < 45$ Gy Constraint')
    ax.legend()

    return ax

  def plot_brain_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Brain', ax, max_dose=72)

    ax.axvline(x=72, color='orange', linestyle='-', label=r'$D_{MAX} < 72$ Gy Constraint')
    ax.legend()

    return ax

  def plot_brainstem_dvh(self, ax=None):
    ax = self.plot_organ_dvh('BrainStem_03', ax, max_dose=54)

    ax.axvline(x=54, color='orange', linestyle='-', label=r'$D_{MAX} < 54$ Gy Constraint')
    ax.legend()

    return ax

  def plot_lungs_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Lungs', ax, max_dose=20)

    ax.axvline(x=20, color='orange', linestyle='--', label=r'$D_{MEAN} < 20$ Gy Constraint')
    ax.legend()

    return ax

  def plot_lens_l_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Lens_L', ax, max_dose=4)

    ax.axvline(x=4, color='orange', linestyle='-', label=r'$D_{MAX} < 4$ Gy Constraint')
    ax.legend()

    return ax

  def plot_lens_r_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Lens_R', ax, max_dose=4)

    ax.axvline(x=4, color='orange', linestyle='-', label=r'$D_{MAX} < 4$ Gy Constraint')
    ax.legend()

    return ax

  def plot_spinal_cord_dvh(self, ax=None):
    ax = self.plot_organ_dvh('SpinalCord_05', ax, max_dose=50)

    ax.axvline(x=50, color='orange', linestyle='-', label=r'$D_{MAX} < 50$ Gy Constraint')
    ax.legend()

    return ax

  def plot_optic_nerve_l_dvh(self, ax=None):
    ax = self.plot_organ_dvh('OpticNerve_L', ax, max_dose=55)

    ax.axvline(x=55, color='orange', linestyle='-', label=r'$D_{MAX} < 55$ Gy Constraint')
    ax.legend()

    return ax

  def plot_optic_nerve_r_dvh(self, ax=None):
    ax = self.plot_organ_dvh('OpticNerve_R', ax, max_dose=55)

    ax.axvline(x=55, color='orange', linestyle='-', label=r'$D_{MAX} < 55$ Gy Constraint')
    ax.legend()

    return ax

  def plot_lips_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Lips', ax, max_dose=50)

    ax.axvline(x=30, color='orange', linestyle='--', label=r'$D_{MEAN} < 30$ Gy Optimal Constraint')
    ax.axvline(x=50, color='orange', linestyle='--', label=r'$D_{MEAN} < 50$ Gy Mandatory Constraint')
    ax.legend()

    return ax

  def plot_chiasm_dvh(self, ax=None):
    ax = self.plot_organ_dvh('Chiasm', ax, max_dose=55)

    ax.axvline(x=55, color='orange', linestyle='-', label=r'$D_{MAX} < 55$ Gy Constraint')
    ax.legend()

    return ax

class ModelResults(BaseResults):
  def __init__(self, model_name, sampling_name, epoch, dataset_split='test'):
    super().__init__(model_name, sampling_name, epoch, dataset_split)

    path = self.base_path / '_sample_metrics.csv'
    self.slice_metrics = pd.read_csv(path)
    self.volume_metrics = self.slice_metrics.groupby('file_name').agg({
      **{col: 'mean' for col in self.slice_metrics.select_dtypes('number').columns},  # mean for all numeric
      'duration': 'sum',
      'slice_index': 'count'
    }).reset_index()

  def for_sample(self, sample):
    return SampleResults(self, sample)



class SampleResults(BaseResults):
  def __init__(self, model_results, sample):
    super().__init__(model_results.model_name, model_results.sampling_name, model_results.epoch, model_results.dataset_split)
    self.sample = sample

    self.slice_metrics = model_results.slice_metrics[model_results.slice_metrics['file_name'] == f'{sample}.npz']
    if len(self.slice_metrics) == 0:
      raise ValueError(f'No data found for sample: {sample}')

    self.volume_metrics = model_results.volume_metrics[model_results.volume_metrics['file_name'] == f'{sample}.npz']


  def for_sample(self, sample):
    if sample != self.sample:
      raise ValueError('SampleResults is already for a specific sample.')
    return self

  def load_orig_volume(self):
    original_idx = {i: slice_index for i, (file, slice_index) in enumerate(self.dataset.index_map) if file.stem == self.sample}

    vol_structure = np.zeros((len(GDPHMM_OAR) + 2, 176, 160, len(original_idx)))
    vol_dose = np.zeros((176, 160, len(original_idx)))

    for i, slice_index in original_idx.items():
      structure, dose, _ = self.dataset[i]
      dose = self.dataset.denormalize_target(dose)[0].numpy()

      vol_structure[:, :, :, slice_index] = structure.numpy()[1:]
      vol_dose[:, :, slice_index] = dose

    return vol_structure, vol_dose


  def load_pred_volume(self):
    original_idx = {i: slice_index for i, (file, slice_index) in enumerate(self.dataset.index_map) if file.stem == self.sample}
    vol_pred_dose = np.zeros((176, 160, len(original_idx)))

    for slice_index in original_idx.values():
      vol_pred_dose[:, :, slice_index] = np.load(self.base_path / f'{self.sample}-{slice_index}.npy')[0]

    return vol_pred_dose


  def plot_volume_dvh(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    structure, original_dose = self.load_orig_volume()
    pred_dose = self.load_pred_volume()

    bins = np.linspace(0, np.max(original_dose), 200)
    colors = cycle(plt.cm.tab10.colors)
    handles, labels = [], []
    for name, mask in zip(['PTV'] + GDPHMM_OAR, structure):
      if mask.sum() == 0:
        continue

      color = next(colors)
      x, y_orig = compute_dvh(original_dose, mask, bins)
      _, y_pred = compute_dvh(pred_dose, mask, bins)
      ax.plot(x, y_orig, color=color, linestyle='-')
      ax.plot(x, y_pred, color=color, linestyle='--')
      handles.append(plt.Line2D([0], [0], color=color, lw=2))
      labels.append(name)

    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume Fraction ≥ Dose')
    ax.set_title('DVH Comparison')
    ax.grid(True, linestyle=':')
    ax.legend(handles, labels, fontsize=8, title='Structures', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=10)

    return ax

class ModelComparison:
  def __init__(self, results_list: List[ModelResults], result_labels=None):
    self.results_list = results_list
    self.result_labels = result_labels if result_labels is not None else [f'{r.model_name} ({r.sampling_name})' for r in results_list]

  def plot_metric_distribution(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = []
    labels = []
    for result, label in zip(self.results_list, self.result_labels):
      data.extend(result.volume_metrics[metric].tolist())
      labels.extend([label] * len(result.volume_metrics[metric]))

    df = pd.DataFrame({'model': labels, 'value': data})
    sns.boxplot(x='model', y='value', data=df, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title(f'Distribution of {METRIC_ABBRS.get(metric, metric)} Across Models')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))

    return ax

  def plot_metric_distributions(self):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, metric in enumerate(DEFAULT_METRICS):
      self.plot_metric_distribution(metric, ax=axes[i])

    fig.suptitle('Metric Distributions Across Models', fontsize=16)
    fig.tight_layout()

    return fig, axes

  def plot_inference_times(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    data = []
    labels = []
    for result, label in zip(self.results_list, self.result_labels):
      data.extend(result.volume_metrics['duration'].tolist())
      labels.extend([label] * len(result.volume_metrics['duration']))

    df = pd.DataFrame({'model': labels, 'inference_time': data})
    sns.boxplot(x='model', y='inference_time', data=df, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Inference Time Distribution Across Models')
    ax.set_ylabel('Inference Time (s)')

    return ax
