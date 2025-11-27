import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from itertools import cycle
from pathlib import Path

from data import create_gdphmm_dataset, GDPHMM_OAR, GDPHMM_DOSE_MAX


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
  'mse_body': 'Mean Squared Error (MSE) Body',
  'rmse_body': 'Root Mean Squared Error (RMSE) Body',
  'psnr_body': 'Peak Signal-to-Noise Ratio (PSNR) Body',
  'dose_score_body': 'Dose Score (MAE) Body',
  'var_laplacian_diff': 'Sharpness Metric (Variance of Laplacian Difference)',
  'mae_challenge': 'Mean Absolute Error (MAE) Challenge',
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
  'mse_body': 'MSE Body',
  'rmse_body': 'RMSE Body',
  'psnr_body': 'PSNR Body',
  'dose_score_body': 'Dose Score (MAE) Body',
  'var_laplacian_diff': 'Sharpness',
  'mae_challenge': 'MAE Challenge',
}

DEFAULT_METRICS = [
  'rmse', 'dose_score', 'dvh',
  'var_laplacian_diff', 'dose_score_body', 'homogeneity_index',
  'ssim', 'mae_challenge', 'psnr_body'
]

HIGHER_IS_BETTER = ['psnr', 'ssim', 'ms_ssim', 'psnr_body', 'var_laplacian_diff']

def build_dataset(dataset_split='test'):
  # Create a config dict for the factory function
  config = {
    'data': {
      'dataset_strategy': 'original',  # Use original dataset for backwards compatibility
      'dataset_name': 'gdphmm-hn-vmat',
      'img_size': [176, 160],
    }
  }

  return create_gdphmm_dataset(config, dataset_split)


def compute_dvh(dose, mask, bins):
  """Compute differential DVH (volume fraction ≥ dose)."""
  masked_dose = dose[mask > 0] if mask is not None else dose

  if masked_dose.size == 0:
    return bins[:-1], np.zeros_like(bins[:-1])

  hist, bin_edges = np.histogram(masked_dose, bins=bins)
  cdf = np.cumsum(hist[::-1])[::-1] / np.sum(hist)
  return bin_edges[:-1], cdf

class BaseResults:
  def __init__(self, model_name, sampling_name, epoch, dataset_split, dataset):
    self.model_name = model_name
    self.sampling_name = sampling_name
    self.epoch = epoch
    self.dataset_split = dataset_split

    self.base_path = SAMPLES_PATH / model_name / sampling_name / f'{dataset_split}_{epoch}'

    self.dataset = dataset

  def plot_quality_over_slice_index(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot one line per result in results_list
    for result, label in zip(self.results_list, self.result_labels):
      grouped = result.slice_metrics.groupby('slice_index')[metric]
      means = grouped.mean()
      stds = grouped.std()

      ax.errorbar(means.index, means.values, yerr=stds.values, fmt='o-', capsize=3, label=label, alpha=0.7)

    dir_symbol = '↑' if metric in HIGHER_IS_BETTER else '↓'
    ax.set_title(f'{METRIC_ABBRS.get(metric, metric)} vs Slice Index (Mean ± SD) \n{dir_symbol} is better')
    ax.set_xlabel('Slice Index')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))

    # Only show legend if there are multiple results
    if len(self.results_list) > 1:
      ax.legend()

    return ax

  def plot_dvh(self, structure_index, ax=None, max_dose=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    bins = np.linspace(0, GDPHMM_DOSE_MAX, 200)
    colors = cycle(plt.cm.tab10.colors)

    # Pre-load all original volumes once (common across all models)
    first_result = self.results_list[0]
    orig_volumes_cache = {}  # file_name -> (structure, original_dose)

    y_orig_all = []
    y_orig_means = []
    for file_name in first_result.slice_metrics['file_name'].unique():
      sample_results = first_result.for_sample(file_name.replace('.npz', ''))
      structure, original_dose = sample_results.load_orig_volume()
      orig_volumes_cache[file_name] = (structure, original_dose)

      mask = structure[structure_index]
      x, y_orig = compute_dvh(original_dose, mask, bins)
      y_orig_all.append(y_orig)
      y_orig_means.append(np.mean(original_dose[mask > 0]))

    if len(y_orig_all) > 0:
      y_orig = np.mean(y_orig_all, axis=0)
      ax.plot(x, y_orig, color='blue', linestyle='-', label='Original Dose', linewidth=2)
      orig_mean = np.mean(y_orig_means)
      ax.axvline(x=orig_mean, color='blue', linestyle='--', alpha=0.7)

      if len(y_orig_means) > 1:
        orig_min, orig_max = np.min(y_orig_means), np.max(y_orig_means)
        ax.axvspan(orig_min, orig_max, color='blue', alpha=0.1)

    # Plot each model's prediction (reuse cached structures)
    for result, label in zip(self.results_list, self.result_labels):
      color = next(colors)
      y_pred_all = []
      y_pred_means = []

      for file_name in result.slice_metrics['file_name'].unique():
        sample_results = result.for_sample(file_name.replace('.npz', ''))
        structure = orig_volumes_cache[file_name][0]  # Reuse cached structure
        pred_dose = sample_results.load_pred_volume()

        mask = structure[structure_index]
        _, y_pred = compute_dvh(pred_dose, mask, bins)
        y_pred_all.append(y_pred)
        y_pred_means.append(np.mean(pred_dose[mask > 0]))

      if len(y_pred_all) > 0:
        y_pred = np.mean(y_pred_all, axis=0)
        ax.plot(x, y_pred, color=color, linestyle='-', label=label)
        pred_mean = np.mean(y_pred_means)
        ax.axvline(x=pred_mean, color=color, linestyle='--', alpha=0.5)

        if len(y_pred_means) > 1:
          pred_min, pred_max = np.min(y_pred_means), np.max(y_pred_means)
          ax.axvspan(pred_min, pred_max, color=color, alpha=0.1)

    ax.set_xlabel('Dose (Gy)')
    ax.set_ylabel('Volume Fraction ≥ Dose')
    ax.grid(True, linestyle=':')
    ax.legend()

    return ax

  def plot_organ_dvh(self, organ_name, ax=None, max_dose=None):
    structure_index = self.dataset.selector.get_oar_index(organ_name)
    ax = self.plot_dvh(structure_index, ax, max_dose)
    ax.set_title(f'{organ_name} DVH Comparison')
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

  def plot_ptv_dvh(self, ax=None):
    # For DVH plotting, we still need the index
    ptv_start, _ = self.dataset.selector.get_channel_indices('ptv')
    ax = self.plot_dvh(ptv_start, ax)
    ax.set_title(f'PTV DVH Comparison')
    return ax


class ModelResults(BaseResults):
  def __init__(self, model_name, sampling_name, epoch, dataset_split, dataset):
    super().__init__(model_name, sampling_name, epoch, dataset_split, dataset)

    path = self.base_path / '_sample_metrics.csv'
    self.slice_metrics = pd.read_csv(path)
    self.volume_metrics = self.slice_metrics.groupby('file_name').agg({
      **{col: 'mean' for col in self.slice_metrics.select_dtypes('number').columns},  # mean for all numeric
      'duration': 'sum',
      'slice_index': 'count'
    }).reset_index()

    # For uniform interface with ModelComparison
    self.results_list = [self]
    self.result_labels = [f'{model_name} ({sampling_name})']

  def for_sample(self, sample):
    return SampleResults(self, sample)



class SampleResults(BaseResults):
  def __init__(self, model_results, sample):
    super().__init__(
      model_results.model_name,
      model_results.sampling_name,
      model_results.epoch,
      model_results.dataset_split,
      model_results.dataset
    )
    self.sample = sample

    self.slice_metrics = model_results.slice_metrics[model_results.slice_metrics['file_name'] == f'{sample}.npz']
    if len(self.slice_metrics) == 0:
      raise ValueError(f'No data found for sample: {sample}')

    self.volume_metrics = model_results.volume_metrics[model_results.volume_metrics['file_name'] == f'{sample}.npz']

    # For uniform interface with ModelComparison
    self.results_list = [self]
    self.result_labels = [f'{model_results.model_name} ({model_results.sampling_name})']

  def for_sample(self, sample):
    if sample != self.sample:
      raise ValueError('SampleResults is already for a specific sample.')
    return self

  def load_orig_volume(self):
    original_idx = {i: slice_index for i, (file, slice_index) in enumerate(self.dataset.index_map) if file.stem == self.sample}

    vol_structure = np.zeros((self.dataset.n_structures, 176, 160, len(original_idx)))
    vol_dose = np.zeros((176, 160, len(original_idx)))

    for i, slice_index in original_idx.items():
      structure, dose, _ = self.dataset[i]
      dose = self.dataset.denormalize_target(dose)[0].numpy()

      vol_structure[:, :, :, slice_index] = structure.numpy()
      vol_dose[:, :, slice_index] = dose

    return vol_structure, vol_dose


  def load_pred_volume(self):
    return np.load(self.base_path / f'{self.sample}.npy')[0]


  def plot_volume_dvh(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    structure, original_dose = self.load_orig_volume()
    pred_dose = self.load_pred_volume()

    bins = np.linspace(0, np.max(original_dose), 200)
    colors = cycle(plt.cm.tab10.colors)
    handles, labels = [], []
    for name, mask in zip(['PTV'] + GDPHMM_OAR, structure[1:]):
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
    ax.legend(handles, labels, fontsize=8, title='Structures', loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=8)

    return ax

  def visualise_slice(self, slice_index, axis='sagittal', ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    if axis == "sagittal":
      original_idx = {i: slice_index for i, (file, slice_index) in enumerate(self.dataset.index_map) if file.stem == self.sample}
      if slice_index not in original_idx.values():
        raise ValueError(f'Slice index {slice_index} not found for sample {self.sample}.')

      idx = [i for i, s_idx in original_idx.items() if s_idx == slice_index][0]
      structure, original_dose, _ = self.dataset[idx]
      original_dose = self.dataset.denormalize_target(original_dose)[0].numpy()
      pred_dose_vol = self.load_pred_volume()
      pred_dose = pred_dose_vol[:, :, slice_index]
    elif axis == "axial" or axis == "coronal":
      structure_vol, original_dose_vol = self.load_orig_volume()
      pred_dose_vol = self.load_pred_volume()

      if axis == "axial":
        structure = structure_vol[:, slice_index, :, :]
        original_dose = original_dose_vol[slice_index, :, :]
        pred_dose = pred_dose_vol[slice_index, :, :]
      else:  # coronal
        structure = np.flip(structure_vol[:, :, slice_index, :], 1)
        original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
        pred_dose = np.flip(pred_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV using selector
    ct_slice = self.dataset.selector.get_ct(structure)[0]
    ptv_mask = self.dataset.selector.get_ptv(structure)[0] > 0

    ax[0].imshow(ct_slice, cmap='gray')
    overlay = np.zeros((*ptv_mask.shape, 4))
    overlay[ptv_mask, 0] = 1.0     # red channel
    overlay[ptv_mask, 3] = 0.3     # alpha channel
    ax[0].imshow(overlay)
    ax[0].contour(ptv_mask, colors='red', linewidths=0.5)
    ax[0].set_title('CT Slice')
    ax[0].axis('off')

    im1 = ax[1].imshow(original_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    ax[1].set_title('Original Dose')
    ax[1].axis('off')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(pred_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    ax[2].set_title('Predicted Dose')
    ax[2].axis('off')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(original_dose - pred_dose, cmap='bwr', vmin=-np.max(np.abs(original_dose - pred_dose)), vmax=np.max(np.abs(original_dose - pred_dose)))
    ax[3].set_title('Error (original - predicted)')
    ax[3].axis('off')
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    return ax

class ModelComparison(BaseResults):
  def __init__(self, results_list: List[ModelResults], result_labels=None):
    if len(results_list) == 0:
      raise ValueError('results_list must contain at least one ModelResults')

    # Use the first result's attributes for BaseResults
    first_result = results_list[0]
    super().__init__(
      model_name=first_result.model_name,
      sampling_name=first_result.sampling_name,
      epoch=first_result.epoch,
      dataset_split=first_result.dataset_split,
      dataset=first_result.dataset
    )

    self.results_list = results_list
    self.result_labels = result_labels if result_labels is not None else [f'{r.model_name} ({r.sampling_name})' for r in results_list]

    # Aggregate slice_metrics and volume_metrics from all results
    self.slice_metrics = pd.concat([r.slice_metrics for r in results_list], ignore_index=True)
    self.volume_metrics = pd.concat([r.volume_metrics for r in results_list], ignore_index=True)

  def for_sample(self, sample):
    # For ModelComparison, we need to handle this differently
    # We could either:
    # 1. Return the first result's sample (for compatibility with single-model code)
    # 2. Raise an error
    # 3. Return a comparison object with all models' samples
    # For now, raise an error to make the intent clear
    raise NotImplementedError(
      'ModelComparison.for_sample() is not directly supported. '
      'Use the comparison object directly for plotting across models, '
      'or access individual results via results_list[i].for_sample(sample)'
    )

  def plot_metric_distribution(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = []
    labels = []
    for result, label in zip(self.results_list, self.result_labels):
      if metric not in result.volume_metrics.columns:
        print(f'Warning: Metric {metric} not found in result for model {label}. Skipping.')
        continue
      data.extend(result.volume_metrics[metric].tolist())
      labels.extend([label] * len(result.volume_metrics[metric]))

    df = pd.DataFrame({'model': labels, 'value': data})
    sns.boxplot(x='model', y='value', data=df, ax=ax)
    ax.tick_params(axis='x', rotation=90)

    dir_symbol = '↑' if metric in HIGHER_IS_BETTER else '↓'
    ax.set_title(f'Distribution of {METRIC_ABBRS.get(metric, metric)} Across Models \n{dir_symbol} is better')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))
    ax.set_xlabel('')

    return ax

  def plot_metric_distributions(self):
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
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
      data.extend(result.volume_metrics['duration'] / 60)
      labels.extend([label] * len(result.volume_metrics['duration']))

    df = pd.DataFrame({'model': labels, 'inference_time': data})
    sns.boxplot(x='model', y='inference_time', data=df, ax=ax)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Inference Time Distribution Across Models')
    ax.set_ylabel('Inference Time (minutes)')
    ax.set_xlabel('')

    return ax

  def plot_quality_over_time_by_volume(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    file_names = self.results_list[0].volume_metrics['file_name'].unique()

    for file_name in file_names:
      points = []
      for result in self.results_list:
        df = result.volume_metrics
        row = df[df['file_name'] == file_name]
        if not row.empty:
          x = row['duration'].iloc[0] / 60
          y = row[metric].iloc[0]
          points.append((x, y))

      points.sort(key=lambda p: p[0])
      if len(points) > 1:
        ax.plot(
          [p[0] for p in points],
          [p[1] for p in points],
          color='gray',
          alpha=0.3,
          linewidth=0.8,
        )

    for result, label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics
      ax.scatter(df['duration'] / 60, df[metric], label=label, alpha=0.8)

    dir_symbol = '↑' if metric in HIGHER_IS_BETTER else '↓'
    ax.set_title(f'{METRIC_ABBRS.get(metric, metric)} vs Inference Time Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Inference Time (minutes)')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))
    ax.legend()

    return ax

  def plot_quality_over_time(self, metric, ax=None):
    # scatter plot with mean values per model
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for result, label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics
      x = df['duration'].mean() / 60
      y = df[metric].mean()
      ax.scatter(x, y, label=label, s=100, alpha=0.6)

    dir_symbol = '↑' if metric in HIGHER_IS_BETTER else '↓'
    ax.set_title(f'Mean {METRIC_ABBRS.get(metric, metric)} vs Mean Inference Time Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Mean Inference Time (minutes)')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))
    ax.legend()

    return ax

  def plot_quality_over_T(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = {}
    for result, result_label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics
      T = int(result.sampling_name.split('-')[-1])
      x = T
      y = df[metric].mean()
      y_std = df[metric].std()
      label = result_label.replace(f' T={T}', '')
      data[label] = data.get(label, []) + [(x, y, y_std)]

    for label, points in data.items():
      points.sort(key=lambda p: p[0])
      ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        marker='o',
        label=label,
      )
      ax.fill_between(
        [p[0] for p in points],
        [p[1] - p[2] for p in points],
        [p[1] + p[2] for p in points],
        alpha=0.2,
      )

    dir_symbol = '↑' if metric in HIGHER_IS_BETTER else '↓'
    ax.set_title(f'Mean (±std) {METRIC_ABBRS.get(metric, metric)} vs Sampling T Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Sampling T')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))
    ax.legend()

    return ax

  def visualise_uncertainty(self, sample, slice_index, ax=None):
    if ax is None:
      fig, ax = plt.subplots(2, 3, figsize=(16, 10))

    preds = []
    for result in self.results_list:
      sample_results = result.for_sample(sample)
      pred_dose_vol = sample_results.load_pred_volume()
      preds.append(pred_dose_vol[:, :, slice_index])

    mean_pred = np.mean(preds, axis=0)
    std_pred = np.std(preds, axis=0)

    sample_results = self.results_list[0].for_sample(sample)
    _, original_dose = sample_results.load_orig_volume()
    original_dose = original_dose[:, :, slice_index]

    im0 = ax[0, 0].imshow(original_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    ax[0, 0].set_title('Original Dose')
    ax[0, 0].axis('off')
    plt.colorbar(im0, ax=ax[0, 0], fraction=0.046, pad=0.04)

    im1 = ax[0, 1].imshow(mean_pred, cmap='gray', vmin=0, vmax=original_dose.max())
    ax[0, 1].set_title('Mean Predicted Dose')
    ax[0, 1].axis('off')
    plt.colorbar(im1, ax=ax[0, 1], fraction=0.046, pad=0.04)

    im2 = ax[0, 2].imshow(std_pred, cmap='hot')
    ax[0, 2].set_title('Prediction Uncertainty (Std Dev)')
    ax[0, 2].axis('off')
    plt.colorbar(im2, ax=ax[0, 2], fraction=0.046, pad=0.04)

    for i, pred in zip(range(3), preds):
      im = ax[1, i].imshow(pred, cmap='gray', vmin=0, vmax=original_dose.max())
      ax[1, i].set_title(f'Predicted Dose - Model {self.result_labels[i]}')
      ax[1, i].axis('off')
      plt.colorbar(im, ax=ax[1, i], fraction=0.046, pad=0.04)

    if fig is not None:
      fig.tight_layout()

    return ax

  def plot_metric_uncertainty(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = []
    sampling_names = []
    labels = []
    for result, result_label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics.copy()
      df['seed'] = result.sampling_name.split('-')[-2]  # assuming sampling_name format includes seed

      sampling_name = result.sampling_name.replace(f'-{df["seed"].iloc[0]}', '')
      df['sampling_name'] = sampling_name

      data.append(df)
      sampling_names.append(sampling_name)
      labels.append(result_label.split(' ')[0])

    metric_std = pd.concat(data).groupby(['file_name', 'sampling_name'], as_index=False)[metric].std()
    sns.boxplot(data=metric_std, x='sampling_name', order=sampling_names, y=metric, ax=ax)
    ax.tick_params(axis='x', rotation=90)

    ax.set_title(f'Uncertainty in {METRIC_ABBRS.get(metric, metric)} Across Models')
    ax.set_ylabel(METRC_LABELS.get(metric, metric))
    ax.set_xlabel('')
    ax.set_xticks(sampling_names, labels)

    return ax

  def viusalise_slice_comparison(self, sample, slice_index, axis='sagittal'):
    fig, axes = plt.subplots(2, len(self.results_list) + 1, figsize=(5 * (len(self.results_list) + 1), 10))

    first_result = self.results_list[0]
    sample_results = first_result.for_sample(sample)
    if axis == "sagittal":
      original_idx = {i: slice_index for i, (file, slice_index) in enumerate(first_result.dataset.index_map) if file.stem == sample}
      if slice_index not in original_idx.values():
        raise ValueError(f'Slice index {slice_index} not found for sample {sample}.')

      idx = [i for i, s_idx in original_idx.items() if s_idx == slice_index][0]
      structure, original_dose, _ = first_result.dataset[idx]
      original_dose = first_result.dataset.denormalize_target(original_dose)[0].numpy()
    elif axis == "axial" or axis == "coronal":
      structure_vol, original_dose_vol = sample_results.load_orig_volume()

      if axis == "axial":
        structure = structure_vol[:, slice_index, :, :]
        original_dose = original_dose_vol[slice_index, :, :]
      else:  # coronal
        structure = np.flip(structure_vol[:, :, slice_index, :], 1)
        original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV using selector
    ct_slice = first_result.dataset.selector.get_ct(structure)[0]
    ptv_mask = first_result.dataset.selector.get_ptv(structure)[0] > 0

    axes[1, 0].imshow(ct_slice, cmap='gray')
    overlay = np.zeros((*ptv_mask.shape, 4))
    overlay[ptv_mask, 0] = 1.0     # red channel
    overlay[ptv_mask, 3] = 0.3     # alpha channel
    axes[1, 0].imshow(overlay)
    axes[1, 0].contour(ptv_mask, colors='red', linewidths=0.5)
    axes[1, 0].set_title('CT Slice')
    axes[1, 0].axis('off')

    im1 = axes[0, 0].imshow(original_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    axes[0, 0].set_title('Original Dose')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

    for i, result in enumerate(self.results_list):
      sample_results = result.for_sample(sample)
      pred_dose_vol = sample_results.load_pred_volume()
      if axis == "sagittal":
        pred_dose = pred_dose_vol[:, :, slice_index]
      elif axis == "axial":
        pred_dose = pred_dose_vol[slice_index, :, :]
      else:  # coronal
        pred_dose = np.flip(pred_dose_vol[:, slice_index, :], 0)

      im = axes[0, i + 1].imshow(pred_dose, cmap='gray', vmin=0, vmax=original_dose.max())
      axes[0, i + 1].set_title(f'{self.result_labels[i]} - Predicted Dose')
      axes[0, i + 1].axis('off')
      plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

      im_err = axes[1, i + 1].imshow(
        original_dose - pred_dose,
        cmap='bwr',
        vmin=-np.max(np.abs(original_dose - pred_dose)),
        vmax=np.max(np.abs(original_dose - pred_dose))
      )
      axes[1, i + 1].set_title(f'{self.result_labels[i]} - Error (original - predicted)')
      axes[1, i + 1].axis('off')
      plt.colorbar(im_err, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    if fig is not None:
      fig.tight_layout()

    return fig, axes

def combine_model_comparisons(results: List[ModelComparison], prefixes: List[str]=None) -> ModelComparison:
  combined_results = []
  combined_labels = []

  for i, comparison in enumerate(results):
    prefix = prefixes[i] + ' ' if prefixes is not None else ''
    for result, label in zip(comparison.results_list, comparison.result_labels):
      if result in combined_results:
        continue  # avoid duplicates
      combined_results.append(result)
      combined_labels.append(prefix + label)

  return ModelComparison(combined_results, combined_labels)
