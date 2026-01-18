import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from itertools import cycle
from pathlib import Path

from data import GDPHMM_OAR, GDPHMM_DOSE_MAX, GDPHMM_OAR_V3
from data_legacy import GDPHMMSlicesDataset, LRUCacheBytes
from constraints import (
    ConstraintLibrary, ConstraintEvaluator,
    FractionationType, ConstraintType, ConstraintGrade,
    GDPHMM_TO_CONSTRAINT_ORGAN
)

plt.style.use("ggplot")

SAMPLES_PATH = Path(__file__).parents[1] / 'samples'

METRIC_LABELS = {
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
  'agreement/agreement_all': 'Criteria Agreement',
  'gamma_3_3/pass_rate': 'Gamma Index 3%/3mm Pass Rate',
  'gamma_2_2/pass_rate': 'Gamma Index 2%/2mm Pass Rate',
  'gamma_3_3/pass_rate': 'Gamma Index 3%/3mm Pass Rate',
  'gamma_5_5/pass_rate': 'Gamma Index 5%/5mm Pass Rate',
  'dvh_auc/OAR/diff': 'DVH Area Between Curves for all OARs',
  'mean_dose/high/abs_diff': 'Mean Dose Diff. (High Dose PTV)',
  'homogeneity_index/high/abs_diff': 'Homogeneity Index Diff. (High Dose PTV)',
  'paddick_conformity_index/high/abs_diff': 'Paddick Conformity Index Diff. (High Dose PTV)',
  'gradient_index/high/abs_diff': 'Paddick Gradient Index Diff. (High Dose PTV)',
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
  'agreement/agreement_all': 'Agreement',
  'gamma_5_5/pass_rate': 'Gamma 5%/5mm',
  'gamma_3_3/pass_rate': 'Gamma 3%/3mm',
  'gamma_2_2/pass_rate': 'Gamma 2%/2mm',
  'gamma_1_1/pass_rate': 'Gamma 1%/1mm',
  'dvh_auc/OAR/diff': 'DVH ABC (OAR)',
  'mean_dose/high/abs_diff': 'Mean Dose Diff (PTV)',
  'homogeneity_index/high/abs_diff': 'HI Diff (PTV)',
  'paddick_conformity_index/high/abs_diff': 'PCI Diff (PTV)',
  'gradient_index/high/abs_diff': 'PGI Diff (PTV)',
}

METRICS_DEFAULT = [
  'rmse', 'dose_score', 'dvh',
  'var_laplacian_diff', 'dose_score_body', 'homogeneity_index',
  'ssim', 'mae_challenge', 'psnr_body'
]

METRICS_PRED = [
  'vol_agreement/agreement_all', 'vol_dvh_auc/OAR/diff', 'vol_dvh',
  'vol_mean_dose/high/abs_diff', 'vol_paddick_conformity_index/high/abs_diff', 'vol_gradient_index/high/abs_diff'
]

METRICS_PLAN = [
  'vol_rmse', 'vol_gamma_3_3/pass_rate', 'vol_agreement/agreement_all', 'vol_dvh_auc/OAR/diff',
]

METRICS_PLAN_EXTENDED = METRICS_PLAN + [
  'vol_mean_dose/high/abs_diff',
  'vol_homogeneity_index/high/abs_diff',
  'vol_paddick_conformity_index/high/abs_diff',
  'vol_gradient_index/high/abs_diff'
]

HIGHER_IS_BETTER = ['psnr', 'ssim', 'ms_ssim', 'psnr_body', 'var_laplacian_diff']

# Patterns for volume metrics where higher values are better
HIGHER_IS_BETTER_PATTERNS = [
  'agreement',
  'conformity_index',
  'csr_pred',
  'csr_target',
  'psnr',
  'ssim',
  'ms_ssim',
  'gamma',
]

# Patterns for volume metrics where lower values are better
LOWER_IS_BETTER_PATTERNS = [
  'rmse',
  'mse',
  'mae',
  'dose_score',
  'abs_diff',
  'diff',
  'homogeneity_index',
  'gradient_index',
]

# PTV line styles: solid for PTV70, dashed for PTV63, dotted for PTV56
PTV_LINESTYLES = ['solid', 'dashed', 'dotted']
PTV_LABELS = ['PTV70', 'PTV63', 'PTV56']


def get_metric_label(metric_name: str) -> str:
  """
  Get a human-readable label for a given metric name.

  Args:
    metric_name: Name of the metric (can be simple like 'rmse' or
                  compound like 'vol_agreement/agreement_all')
  Returns:
    A human-readable label for the metric.
  """
  if metric_name.startswith('vol_'):
    metric_name = metric_name[4:]  # Remove 'vol_' prefix

  if metric_name in METRIC_LABELS:
    return METRIC_LABELS[metric_name]

  for key in METRIC_LABELS.keys():
    if key.startswith(metric_name):
      return METRIC_LABELS[key]

  return metric_name

def get_metric_abbr(metric_name: str) -> str:
  """
  Get a human-readable abbreviation for a given metric name.

  Args:
    metric_name: Name of the metric (can be simple like 'rmse' or
                  compound like 'vol_agreement/agreement_all')

  Returns:
    A human-readable abbreviation for the metric.
  """
  if metric_name.startswith('vol_'):
    metric_name = metric_name[4:]  # Remove 'vol_' prefix

  if metric_name in METRIC_ABBRS:
    return METRIC_ABBRS[metric_name]

  for key in METRIC_ABBRS.keys():
    if key.startswith(metric_name):
      return METRIC_ABBRS[key]

  return metric_name

def is_higher_better(metric_name: str) -> bool:
  """
  Check if a metric is 'higher is better' based on its name.

  Args:
    metric_name: Name of the metric (can be simple like 'rmse' or
                  compound like 'vol_agreement/agreement_all')

  Returns:
    True if higher values are better, False if lower values are better.
    Defaults to False (lower is better) for unknown metrics.
  """
  metric_lower = metric_name.lower()

  # Check simple metrics first
  if metric_name in HIGHER_IS_BETTER:
    return True

  # Check patterns for higher is better
  for pattern in HIGHER_IS_BETTER_PATTERNS:
    if pattern in metric_lower:
      # Make sure it's not a diff/abs_diff variant (those are lower is better)
      if 'diff' not in metric_lower or pattern == 'diff':
        return True

  # Default to lower is better (most error metrics)
  return False

def draw_ptv_contours(ax, ptv_doses, color='white', linewidths=0.5):
  """
  Draw contours for all three PTV levels on the given axes.

  Args:
      ax: Matplotlib axes
      ptv_doses: Array of shape (3, H, W) with PTV high, mid, low dose masks
      color: Color for all contours
      linewidths: Line width for contours
  """
  for mask, linestyle in zip(ptv_doses, PTV_LINESTYLES):
    if mask.any():
      ax.contour(mask > 0, colors=[color], linewidths=linewidths, linestyles=[linestyle])

def draw_oar_contours(ax, oar_masks, color='cyan', linewidths=0.5, alpha=0.7):
  """
  Draw contours for all OAR structures on the given axes.

  Args:
      ax: Matplotlib axes
      oar_masks: Array of shape (N, H, W) with OAR masks
      color: Color for all contours
      linewidths: Line width for contours
      alpha: Transparency for contours
  """
  for mask in oar_masks:
    if mask.any():
      ax.contour(mask > 0, colors=[color], linewidths=linewidths, alpha=alpha)

def draw_ptv_overlay(ax, ptv_doses, color='red', alpha=0.3):
  """
  Draw filled overlay for all three PTV levels on the given axes.

  Args:
      ax: Matplotlib axes
      ptv_doses: Array of shape (3, H, W) with PTV high, mid, low dose masks
      color: Color for the overlay
      alpha: Transparency of the overlay
  """
  # Combine all masks into one
  combined_mask = (ptv_doses[0] > 0) | (ptv_doses[1] > 0) | (ptv_doses[2] > 0)
  overlay = np.zeros((*combined_mask.shape, 4))
  # Convert color name to RGB if needed
  if color == 'red':
    rgb = (1.0, 0.0, 0.0)
  elif color == 'blue':
    rgb = (0.0, 0.0, 1.0)
  else:
    rgb = (1.0, 0.0, 0.0)  # default to red
  overlay[combined_mask, 0] = rgb[0]
  overlay[combined_mask, 1] = rgb[1]
  overlay[combined_mask, 2] = rgb[2]
  overlay[combined_mask, 3] = alpha
  ax.imshow(overlay)

def build_dataset(dataset_split='test'):
  cache = LRUCacheBytes(max_bytes=5 * 1024 ** 3)  # 5 GB cache

  # Create the dataset using the legacy class
  dataset = GDPHMMSlicesDataset(
    dataset_name='gdphmm-hn-vmat',
    split=dataset_split,
    img_size=[176, 160],
    cache=cache,
    structure_groups=[
      'ct',
      'oar_v3',
      'ptv_doses',
    ],
    limit=None,
    skip=0,
    voxel_spacing=[2.5, 2.5, 2.0],
    mode='3d',
  )

  return dataset


def compute_dvh(dose, mask, bins):
  """Compute differential DVH (volume fraction ≥ dose)."""
  masked_dose = dose[mask > 0] if mask is not None else dose

  if masked_dose.size == 0:
    return bins[:-1], np.zeros_like(bins[:-1])

  hist, bin_edges = np.histogram(masked_dose, bins=bins)
  cdf = np.cumsum(hist[::-1])[::-1] / np.sum(hist)
  return bin_edges[:-1], cdf


def iqr_limits_from_df(
  df,
  group_col,
  value_col,
  whis=1.5,
  margin=0.05,
):
  """
  Compute global ymin/ymax based on whis * IQR across all groups.

  Parameters
  ----------
  df : pandas.DataFrame
  group_col : str
    Column used for grouping (box categories).
  value_col : str
    Numeric values.
  whis : float, default 1.5
    Whisker length in units of IQR.
  margin : float, default 0.05
    Additional margin to add to ymin/ymax.

  Returns
  -------
  ymin, ymax : float
  """
  whisker_mins = []
  whisker_maxs = []

  for _, g in df.groupby(group_col):
    values = g[value_col].dropna().to_numpy()
    if values.size == 0:
      continue

    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1

    lower_fence = q1 - whis * iqr
    upper_fence = q3 + whis * iqr

    # Actual whiskers = most extreme points inside the fences
    inside = values[(values >= lower_fence) & (values <= upper_fence)]

    if inside.size == 0:
      continue

    whisker_mins.append(inside.min())
    whisker_maxs.append(inside.max())

  if not whisker_mins:
    raise ValueError("No valid data found to compute whisker limits.")

  vmin = min(whisker_mins)
  vmax = max(whisker_maxs)
  margin_value = margin * (vmax - vmin)
  return vmin, vmax, min(whisker_mins) - margin_value, vmax + margin_value


class BaseResults:
  def __init__(self, model_name, sampling_name, epoch, dataset_split, dataset):
    self.model_name = model_name
    self.sampling_name = sampling_name
    self.epoch = epoch
    self.dataset_split = dataset_split

    self.base_path = SAMPLES_PATH / model_name / sampling_name / f'{dataset_split}_{epoch}'

    self.dataset = dataset

    # Initialize constraint library and evaluator
    self.constraint_library = ConstraintLibrary()
    self.constraint_evaluator = ConstraintEvaluator(self.constraint_library.constraints)
    self.fractionation = FractionationType.CONVENTIONAL  # Default fractionation

  def get_organ_constraints(self, organ_name: str, constraint_type: Optional[ConstraintType] = None) -> List:
    """
    Get constraints for a specific organ from the constraint library.

    Args:
        organ_name: Name of the organ from GDPHMM_OAR list
        constraint_type: Optional filter for specific constraint type

    Returns:
        List of constraints for the organ
    """
    # Map GDPHMM organ name to constraint library organ name
    constraint_organ = GDPHMM_TO_CONSTRAINT_ORGAN.get(organ_name)
    if constraint_organ is None:
      return []

    return self.constraint_library.get_constraints(
      organ=constraint_organ,
      fractionation=self.fractionation,
      constraint_type=constraint_type
    )

  def plot_quality_over_slice_index(self, metric, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot one line per result in results_list
    for result, label in zip(self.results_list, self.result_labels):
      grouped = result.slice_metrics.groupby('slice_index')[metric]
      means = grouped.mean()
      stds = grouped.std()

      ax.errorbar(means.index, means.values, yerr=stds.values, fmt='o-', capsize=3, label=label, alpha=0.7)

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(f'{get_metric_abbr(metric)} vs Slice Index (Mean ± SD) \n{dir_symbol} is better')
    ax.set_xlabel('Slice Index')
    ax.set_ylabel(get_metric_label(metric))

    # Only show legend if there are multiple results
    if len(self.results_list) > 1:
      ax.legend()

    return ax

  def plot_dvh(self, structure_index, ax=None, max_dose=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    bins = np.linspace(0, GDPHMM_DOSE_MAX, 200)
    colors = cycle(plt.cm.tab10.colors)

    first_result = self.results_list[0]
    file_names = first_result.slice_metrics['file_name'].unique()

    # Pre-load masks and compute original DVH in one pass (don't store full dose volumes)
    cached_masks = {}  # {sample_name: mask}
    y_orig_all = []
    y_orig_means = []

    for file_name in file_names:
      sample_name = file_name.replace('.npz', '')
      sample_results = first_result.for_sample(sample_name)
      structure, original_dose = sample_results.load_orig_volume()
      mask = structure[structure_index] > 0
      cached_masks[sample_name] = mask

      # Compute original DVH immediately, then discard original_dose
      x, y_orig = compute_dvh(original_dose, mask, bins)
      y_orig_all.append(y_orig)
      if mask.sum() > 0:
        y_orig_means.append(np.mean(original_dose[mask > 0]))

    if len(y_orig_all) > 0:
      y_orig = np.mean(y_orig_all, axis=0)
      ax.plot(x, y_orig, color='blue', linestyle='-', label='Original Dose', linewidth=2)
      if len(y_orig_means) > 0:
        orig_mean = np.mean(y_orig_means)
        ax.axvline(x=orig_mean, color='blue', linestyle='--', alpha=0.7)

        if len(y_orig_means) > 1:
          orig_min, orig_max = np.min(y_orig_means), np.max(y_orig_means)
          ax.axvspan(orig_min, orig_max, color='blue', alpha=0.1)

    # Plot each model's prediction using cached masks
    for result, label in zip(self.results_list, self.result_labels):
      color = next(colors)
      y_pred_all = []
      y_pred_means = []

      for file_name in result.slice_metrics['file_name'].unique():
        sample_name = file_name.replace('.npz', '')
        mask = cached_masks[sample_name]
        # Load prediction from the current result's path
        pred_dose = np.load(result.base_path / f'{sample_name}.npy')[0]

        _, y_pred = compute_dvh(pred_dose, mask, bins)
        y_pred_all.append(y_pred)
        if mask.sum() > 0:
          y_pred_means.append(np.mean(pred_dose[mask > 0]))

      if len(y_pred_all) > 0:
        y_pred = np.mean(y_pred_all, axis=0)
        ax.plot(x, y_pred, color=color, linestyle='-', label=label)
        if len(y_pred_means) > 0:
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

  def plot_organ_dvh(self, organ_name, ax=None, max_dose=None, show_constraints=True):
    """
    Plot DVH for a specific organ with optional constraint visualization.

    Args:
        organ_name: Name of the organ from GDPHMM_OAR list
        ax: Matplotlib axes to plot on
        max_dose: Optional maximum dose for x-axis
        show_constraints: Whether to show constraint lines from library

    Returns:
        Matplotlib axes
    """
    structure_index = self.dataset.selector.get_oar_index(organ_name)
    ax = self.plot_dvh(structure_index, ax, max_dose)
    ax.set_title(f'{organ_name} DVH Comparison')

    # Add constraint lines from library
    if show_constraints:
      constraints = self.get_organ_constraints(organ_name)
      colors = ['orange', 'red', 'darkred', 'purple']
      linestyles = {
        ConstraintGrade.OPTIMAL: ':',
        ConstraintGrade.RECOMMENDED: '--',
        ConstraintGrade.MANDATORY: '-',
        ConstraintGrade.ACCEPTABLE: '-.',
        None: '-'
      }

      for i, constraint in enumerate(constraints):
        color = colors[i % len(colors)]
        linestyle = linestyles.get(constraint.grade, '-')

        # Only plot D_MAX and D_MEAN constraints as vertical lines
        if constraint.constraint_type == ConstraintType.D_MAX:
          label = f'D_MAX {constraint.comparison} {constraint.limit_value} Gy'
          if constraint.grade:
            label += f' ({constraint.grade.value})'
          ax.axvline(x=constraint.limit_value, color=color, linestyle=linestyle, alpha=0.7, label=label)

        elif constraint.constraint_type == ConstraintType.D_MEAN:
          label = f'D_MEAN {constraint.comparison} {constraint.limit_value} Gy'
          if constraint.grade:
            label += f' ({constraint.grade.value})'
          ax.axvline(x=constraint.limit_value, color=color, linestyle=linestyle, alpha=0.7, label=label)

      ax.legend()

    return ax

  def plot_eye_dvh(self, ax=None):
    return self.plot_organ_dvh('Eyes', ax, max_dose=50)

  def plot_cochlea_l_dvh(self, ax=None):
    return self.plot_organ_dvh('Cochlea_L', ax, max_dose=50)

  def plot_cochlea_r_dvh(self, ax=None):
    return self.plot_organ_dvh('Cochlea_R', ax, max_dose=50)

  def plot_brain_dvh(self, ax=None):
    return self.plot_organ_dvh('Brain', ax, max_dose=80)

  def plot_brainstem_dvh(self, ax=None):
    return self.plot_organ_dvh('BrainStem_03', ax, max_dose=60)

  def plot_lungs_dvh(self, ax=None):
    return self.plot_organ_dvh('Lungs', ax, max_dose=50)

  def plot_lens_l_dvh(self, ax=None):
    return self.plot_organ_dvh('Lens_L', ax, max_dose=10)

  def plot_lens_r_dvh(self, ax=None):
    return self.plot_organ_dvh('Lens_R', ax, max_dose=10)

  def plot_spinal_cord_dvh(self, ax=None):
    return self.plot_organ_dvh('SpinalCord_05', ax, max_dose=60)

  def plot_optic_nerve_l_dvh(self, ax=None):
    return self.plot_organ_dvh('OpticNerve_L', ax, max_dose=60)

  def plot_optic_nerve_r_dvh(self, ax=None):
    return self.plot_organ_dvh('OpticNerve_R', ax, max_dose=60)

  def plot_lips_dvh(self, ax=None):
    return self.plot_organ_dvh('Lips', ax, max_dose=60)

  def plot_chiasm_dvh(self, ax=None):
    return self.plot_organ_dvh('Chiasm', ax, max_dose=60)

  def plot_ptv_dvh(self, ax=None):
    # For DVH plotting, we still need the index
    ptv_start, _ = self.dataset.selector.get_channel_indices('ptv_doses')
    ax = self.plot_dvh(ptv_start, ax)
    ax.set_title(f'PTV DVH Comparison')
    return ax


class ModelResults(BaseResults):
  def __init__(self, model_name, sampling_name, epoch, dataset_split, dataset):
    super().__init__(model_name, sampling_name, epoch, dataset_split, dataset)

    path = self.base_path / '_sample_metrics.csv'
    self.slice_metrics = pd.read_csv(path)

    volume_path = self.base_path / '_volume_metrics.csv'
    self.volume_metrics = pd.read_csv(volume_path).set_index('file_name')
    self.volume_metrics['duration'] = self.slice_metrics.groupby(self.slice_metrics['file_name'].str.replace('.npz', ''))['duration'].sum().values
    self.volume_metrics = self.volume_metrics.reset_index()

    self.volume_metrics['vol_rmse/OAR'] = self.volume_metrics[[f'vol_rmse/{organ}' for organ in GDPHMM_OAR_V3]].mean(axis=1)
    self.volume_metrics['vol_mse/OAR'] = self.volume_metrics[[f'vol_mse/{organ}' for organ in GDPHMM_OAR_V3]].mean(axis=1)
    self.volume_metrics['vol_dose_score/OAR'] = self.volume_metrics[[f'vol_dose_score/{organ}' for organ in GDPHMM_OAR_V3]].mean(axis=1)
    self.volume_metrics['vol_dvh/OAR'] = self.volume_metrics[[f'vol_dvh/{organ}' for organ in GDPHMM_OAR_V3]].mean(axis=1)
    self.volume_metrics['vol_dvh_auc/OAR/diff'] = self.volume_metrics[[f'vol_dvh_auc/{organ}/diff' for organ in GDPHMM_OAR_V3]].mean(axis=1)

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
    original_idx = [i for i, (file, _) in enumerate(self.dataset.index_map) if file.stem == self.sample][0]

    structure, dose, _ = self.dataset[original_idx]
    structure = structure.numpy()
    dose = self.dataset.denormalize_target(dose)[0].numpy()

    return structure, dose


  def load_pred_volume(self):
    return np.load(self.base_path / f'{self.sample}.npy')[0]


  def plot_volume_dvh(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    structure, original_dose = self.load_orig_volume()
    pred_dose = self.load_pred_volume()

    structure = structure[np.newaxis, ...]  # Add batch dimension
    ptv_masks = self.dataset.selector.get_ptv_doses(structure) > 0
    oar_masks = self.dataset.selector.get_component(structure, 'oar_v3') > 0

    print('ptv_masks', ptv_masks.shape)
    print('oar_masks', oar_masks.shape)
    masks = np.concatenate([ptv_masks[:, 0:1], oar_masks], axis=1)[0]
    print('masks', masks.shape)

    bins = np.linspace(0, np.max(original_dose), 200)
    colors = cycle(plt.cm.tab10.colors)
    handles, labels = [], []
    for name, mask in zip(['PTV'] + GDPHMM_OAR_V3, masks):
      if mask.sum() == 0:
        continue

      color = next(colors)
      x, y_orig = compute_dvh(original_dose, mask, bins)
      _, y_pred = compute_dvh(pred_dose, mask, bins)
      ax.plot(x, y_orig, color=color, linestyle='-')
      # ax.plot(x, y_pred, color=color, linestyle='--')
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

    structure_vol, original_dose_vol = self.load_orig_volume()
    pred_dose_vol = self.load_pred_volume()

    if axis == "sagittal":
      structure = structure_vol[:, :, :, slice_index]
      original_dose = original_dose_vol[:, :, slice_index]
      pred_dose = pred_dose_vol[:, :, slice_index]
    elif axis == "axial":
      structure = structure_vol[:, slice_index, :, :]
      original_dose = original_dose_vol[slice_index, :, :]
      pred_dose = pred_dose_vol[slice_index, :, :]
    elif axis == "coronal":
      structure = np.flip(structure_vol[:, :, slice_index, :], 1)
      original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
      pred_dose = np.flip(pred_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV masks using selector
    ct_slice = self.dataset.selector.get_ct(structure)[0]
    ptv_doses = self.dataset.selector.get_component(structure, 'ptv_doses')

    ax[0].imshow(ct_slice, cmap='gray')
    draw_ptv_overlay(ax[0], ptv_doses)
    draw_ptv_contours(ax[0], ptv_doses)
    ax[0].set_title('CT Slice')
    ax[0].axis('off')

    im1 = ax[1].imshow(original_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    draw_ptv_contours(ax[1], ptv_doses)
    ax[1].set_title('Original Dose')
    ax[1].axis('off')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04)

    im2 = ax[2].imshow(pred_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    draw_ptv_contours(ax[2], ptv_doses)
    ax[2].set_title('Predicted Dose')
    ax[2].axis('off')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04)

    im3 = ax[3].imshow(original_dose - pred_dose, cmap='bwr', vmin=-np.max(np.abs(original_dose - pred_dose)), vmax=np.max(np.abs(original_dose - pred_dose)))
    draw_ptv_contours(ax[3], ptv_doses)
    ax[3].set_title('Error (original - predicted)')
    ax[3].axis('off')
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04)

    return ax

  def visualise_slice_with_ct(self, slice_index, axis='sagittal', ax=None, dose_cmap='jet', dose_alpha=0.5, show_oar=True):
    """
    Visualize a slice with the CT scan underneath transparent dose distributions.

    Args:
        slice_index: Index of the slice to visualize
        axis: 'sagittal', 'axial', or 'coronal'
        ax: Optional matplotlib axes array (expects 4 axes)
        dose_cmap: Colormap for dose overlay (default: 'jet')
        dose_alpha: Transparency of dose overlay (default: 0.5)
        show_oar: Whether to show OAR contours (default: True)

    Returns:
        Matplotlib axes
    """
    if ax is None:
      fig, ax = plt.subplots(1, 4, figsize=(20, 5))

    structure_vol, original_dose_vol = self.load_orig_volume()
    pred_dose_vol = self.load_pred_volume()

    if axis == "sagittal":
      structure = structure_vol[:, :, :, slice_index]
      original_dose = original_dose_vol[:, :, slice_index]
      pred_dose = pred_dose_vol[:, :, slice_index]

      structure = np.flip(structure, [1, 2])
      original_dose = np.flip(original_dose)
      pred_dose = np.flip(pred_dose)
    elif axis == "axial":
      structure = structure_vol[:, slice_index, :, :]
      original_dose = original_dose_vol[slice_index, :, :]
      pred_dose = pred_dose_vol[slice_index, :, :]
    elif axis == "coronal":
      structure = np.flip(structure_vol[:, :, slice_index, :], 1)
      original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
      pred_dose = np.flip(pred_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV masks using selector
    ct_slice = self.dataset.selector.get_ct(structure)[0]
    ptv_doses = self.dataset.selector.get_component(structure, 'ptv_doses')

    # Get OAR masks if needed
    oar_masks = None
    if show_oar:
      oar_masks = self.dataset.selector.get_component(structure, 'oar_v3')

    dose_max = original_dose.max()

    # Panel 0: CT with PTV overlay
    ax[0].imshow(ct_slice, cmap='gray')
    draw_ptv_overlay(ax[0], ptv_doses)
    draw_ptv_contours(ax[0], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(ax[0], oar_masks)
    ax[0].set_title('CT with PTV' + (' & OAR' if show_oar else ''))
    ax[0].axis('off')

    # Panel 1: CT with original dose overlay
    ax[1].imshow(ct_slice, cmap='gray')
    dose_masked = np.ma.masked_where(original_dose < 0.01 * dose_max, original_dose)
    im1 = ax[1].imshow(dose_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=0, vmax=dose_max)
    draw_ptv_contours(ax[1], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(ax[1], oar_masks)
    ax[1].set_title('Original Dose on CT')
    ax[1].axis('off')
    plt.colorbar(im1, ax=ax[1], fraction=0.046, pad=0.04, label='Dose (Gy)')

    # Panel 2: CT with predicted dose overlay
    ax[2].imshow(ct_slice, cmap='gray')
    pred_masked = np.ma.masked_where(pred_dose < 0.01 * dose_max, pred_dose)
    im2 = ax[2].imshow(pred_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=0, vmax=dose_max)
    draw_ptv_contours(ax[2], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(ax[2], oar_masks)
    ax[2].set_title('Predicted Dose on CT')
    ax[2].axis('off')
    plt.colorbar(im2, ax=ax[2], fraction=0.046, pad=0.04, label='Dose (Gy)')

    # Panel 3: Error map with CT underneath
    ax[3].imshow(ct_slice, cmap='gray')
    error = original_dose - pred_dose
    error_max = np.max(np.abs(error))
    error_masked = np.ma.masked_where(np.abs(error) < 0.01 * error_max, error)
    im3 = ax[3].imshow(error_masked, cmap='bwr', alpha=dose_alpha, vmin=-error_max, vmax=error_max)
    draw_ptv_contours(ax[3], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(ax[3], oar_masks)
    ax[3].set_title('Error on CT (original - predicted)')
    ax[3].axis('off')
    plt.colorbar(im3, ax=ax[3], fraction=0.046, pad=0.04, label='Error (Gy)')

    return ax

class ModelComparison(BaseResults):
  def __init__(self, results_list: List[ModelResults], result_labels=None, result_groups=None):
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
    self.has_groups = result_groups is not None
    self.result_groups = result_groups or [''] * len(results_list)

    # Aggregate slice_metrics and volume_metrics from all results
    slice_metrics = []
    volume_metrics = []
    for r, label in zip(results_list, result_labels):
      sm = r.slice_metrics
      sm['label'] = label
      slice_metrics.append(sm)

      vm = r.volume_metrics
      vm['label'] = label
      volume_metrics.append(vm)

    self.slice_metrics = pd.concat(slice_metrics, ignore_index=True)
    self.volume_metrics = pd.concat(volume_metrics, ignore_index=True)

  def __add__(self, other):
    """
    Add ModelResults or another ModelComparison to this comparison.

    Args:
        other: A ModelResults, ModelComparison, or list of ModelResults to add

    Returns:
        A new ModelComparison with the added results
    """
    if isinstance(other, ModelResults):
      to_add = [other]
      labels_to_add = [f'{other.model_name} ({other.sampling_name})']
    elif isinstance(other, ModelComparison):
      to_add = other.results_list
      labels_to_add = other.result_labels
    elif isinstance(other, (list, tuple)):
      to_add = list(other)
      labels_to_add = [f'{r.model_name} ({r.sampling_name})' for r in to_add]
    else:
      raise TypeError(f'Cannot add {type(other).__name__} to ModelComparison')

    new_results = list(self.results_list)
    new_labels = list(self.result_labels)
    for result, label in zip(to_add, labels_to_add):
      if result not in new_results:
        new_results.append(result)
        new_labels.append(label)

    return ModelComparison(new_results, new_labels)

  def __sub__(self, other):
    """
    Exclude specific ModelResults from the comparison.

    Args:
        other: A ModelResults instance or a list of ModelResults to exclude

    Returns:
        A new ModelComparison without the excluded results
    """
    if isinstance(other, ModelResults):
      to_exclude = [other]
    elif isinstance(other, (list, tuple)):
      to_exclude = list(other)
    else:
      raise TypeError(f'Cannot subtract {type(other).__name__} from ModelComparison')

    new_results = []
    new_labels = []
    for result, label in zip(self.results_list, self.result_labels):
      if result not in to_exclude:
        new_results.append(result)
        new_labels.append(label)

    if len(new_results) == 0:
      raise ValueError('Cannot create ModelComparison with no results')

    return ModelComparison(new_results, new_labels)

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

  def plot_metric_distribution(self, metric, ax=None, sort=False, outliers=True):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    data = []
    labels = []
    groups = []
    for result, label, group in zip(self.results_list, self.result_labels, self.result_groups):
      if metric not in result.volume_metrics.columns:
        print(f'Warning: Metric {metric} not found in result for model {label}. Skipping.')
        continue
      data.extend(result.volume_metrics[metric].tolist())
      labels.extend([label] * len(result.volume_metrics[metric]))
      groups.extend([group] * len(result.volume_metrics[metric]))

    df = pd.DataFrame({'model': labels, 'value': data, 'group': groups})

    # Sort by median, then IQR, then range if requested
    if sort:
      higher_is_better = is_higher_better(metric)
      ascending = [not higher_is_better, True, True]
      grouped = df.groupby('model')['value']
      sort_df = pd.DataFrame({
        'median': grouped.median(),
        'iqr': grouped.quantile(0.75) - grouped.quantile(0.25),
        'range': grouped.max() - grouped.min()
      })
      order = sort_df.sort_values(by=['median', 'iqr', 'range'], ascending=ascending).index.tolist()
    else:
      order = None

    medianprops = {'color': 'black', 'linewidth': 2}
    if outliers:
      sns.boxplot(x='model', y='value', data=df, ax=ax, order=order, hue='group', dodge=False, medianprops=medianprops)
    else:
      sns.boxplot(x='model', y='value', data=df, ax=ax, order=order, hue='group', dodge=False, whis=1.5, medianprops=medianprops)
      vmin, vmax, ymin, ymax = iqr_limits_from_df(df, 'model', 'value', whis=1.5)
      ax.set_ylim(ymin, ymax)

      model_order = order if order is not None else df['model'].unique().tolist()
      model_to_x = {model: i for i, model in enumerate(model_order)}

      above_mask = df['value'] > vmax
      below_mask = df['value'] < vmin
      above_x = df.loc[above_mask, 'model'].map(model_to_x)
      below_x = df.loc[below_mask, 'model'].map(model_to_x)
      ax.plot(above_x, np.full(len(above_x), vmax), marker="^", linestyle="none", color="black")
      ax.plot(below_x, np.full(len(below_x), vmin), marker="v", linestyle="none", color="black")

    ax.tick_params(axis='x', rotation=90)

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(f'{get_metric_label(metric)}\n{dir_symbol} is better')
    ax.set_ylabel(get_metric_abbr(metric))
    ax.set_xlabel('')

    if self.has_groups:
      ax.legend(title='')

    return ax

  def plot_metric_distributions(self, metrics=METRICS_DEFAULT, sort=False, outliers=True):
    if len(metrics) % 4 == 0:
      n_cols = 4
      n_rows = len(metrics) // 4
    else:
      n_metrics = len(metrics)
      n_cols = min(3, n_metrics)
      n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows), squeeze=False)
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
      self.plot_metric_distribution(metric, ax=axes[i], sort=sort, outliers=outliers)

    fig.suptitle('Metric Distributions Across Models', fontsize=16)
    fig.tight_layout()

    if self.has_groups:
      handles, labels = axes[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper right', title='', ncol=8)

      for ax in axes:
        ax.get_legend().remove()

    return fig, axes

  def plot_inference_times(self, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    data = []
    labels = []
    groups = []
    for result, label, group in zip(self.results_list, self.result_labels, self.result_groups):
      data.extend(result.volume_metrics['duration'] / 60)
      labels.extend([label] * len(result.volume_metrics['duration']))
      groups.extend([group] * len(result.volume_metrics['duration']))

    df = pd.DataFrame({'model': labels, 'inference_time': data, 'group': groups})
    sns.boxplot(x='model', y='inference_time', data=df, ax=ax, hue='group', dodge=False)
    ax.tick_params(axis='x', rotation=90)
    ax.set_title('Inference Time Distribution Across Models')
    ax.set_ylabel('Inference Time (minutes)')
    ax.set_xlabel('')

    if self.has_groups:
      ax.legend(title='')

    return ax

  def plot_ptv_evaluation(self, ax=None):
    if ax is None:
      fig, axes = plt.subplots(1, 4, figsize=(15, 4))

    axes[0].fill_between([55, 75], [55, 75], 75, color='green', alpha=0.1)
    axes[0].fill_between([55, 75], [55, 75], 55, color='red', alpha=0.1)
    sns.scatterplot(self.volume_metrics, x=f'vol_mean_dose/high/target', y=f'vol_mean_dose/high/pred', style='label', hue='label', ax=axes[0], legend='full')
    #axes[0].set_title(r'Mean Dose $\tilde{x} = ' + str(self.volume_metrics['vol_mean_dose/high/abs_diff'].median().round(2)) + '$')
    axes[0].set_title('Mean Dose')
    axes[0].set_xlabel('Target Mean Dose [Gy]')
    axes[0].set_ylabel('Predicted Mean Dose [Gy]')
    axes[0].set_xlim(55, 75)
    axes[0].set_ylim(55, 75)
    axes[0].xaxis.get_major_locator().set_params(integer=True)
    axes[0].yaxis.get_major_locator().set_params(integer=True)

    axes[1].fill_between([0, 0.4], [0, 0.4], 0.4, color='red', alpha=0.1)
    axes[1].fill_between([0, 0.4], [0, 0.4], 0, color='green', alpha=0.1)
    sns.scatterplot(self.volume_metrics, x=f'vol_homogeneity_index/high/target', y=f'vol_homogeneity_index/high/pred', style='label', hue='label', ax=axes[1])
    axes[1].set_title(f'Homogeneity Index')
    axes[1].set_xlabel('Target Homogeneity Index')
    axes[1].set_ylabel('Predicted Homogeneity Index')
    axes[1].set_xlim(0, 0.40)
    axes[1].set_ylim(0, 0.40)

    axes[2].fill_between([0, 1], [0, 1], 1, color='green', alpha=0.1)
    axes[2].fill_between([0, 1], [0, 1], 0, color='red', alpha=0.1)
    sns.scatterplot(self.volume_metrics, x=f'vol_paddick_conformity_index/high/target', y=f'vol_paddick_conformity_index/high/pred', style='label', hue='label', ax=axes[2])
    axes[2].set_title(f'Paddick Conformity Index')
    axes[2].set_xlabel('Target Conformity Index')
    axes[2].set_ylabel('Predicted Conformity Index')
    axes[2].set_xlim(0, 1.0)
    axes[2].set_ylim(0, 1.0)

    axes[3].fill_between([0, 93], [0, 93], 93, color='red', alpha=0.1)
    axes[3].fill_between([0, 93], [0, 93], 0, color='green', alpha=0.1)
    sns.scatterplot(self.volume_metrics, x=f'vol_gradient_index/high/target', y=f'vol_gradient_index/high/pred', style='label', hue='label', ax=axes[3])
    axes[3].set_title(f'Gradient Index')
    axes[3].set_xlabel('Target Gradient Index')
    axes[3].set_ylabel('Predicted Gradient Index')
    axes[3].set_xlim(0, 93)
    axes[3].set_ylim(0, 93)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', title='', ncol=8, bbox_to_anchor=(0.5, -0.1))

    for ax in axes:
      ax.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
      ax.set_aspect('equal')
      ax.get_legend().remove()
      ax.set_facecolor('white')

    if fig is not None:
      fig.suptitle('PTV Evaluation')
      fig.tight_layout()

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

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(f'{get_metric_abbr(metric)} vs Inference Time Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Inference Time (minutes)')
    ax.set_ylabel(get_metric_label(metric))
    ax.legend()

    return ax

  def plot_quality_over_time(self, metric, ax=None):
    # scatter plot with median values per model
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for result, label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics
      x = df['duration'].median() / 60
      y = df[metric].median()
      ax.scatter(x, y, label=label, s=100, alpha=0.6)

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(f'Median {get_metric_abbr(metric)} vs Median Inference Time Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Median Inference Time (minutes)')
    ax.set_ylabel('Median ' + get_metric_label(metric))
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

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(f'Mean (±std) {get_metric_abbr(metric)} vs Sampling T Across Models \n{dir_symbol} is better')
    ax.set_xlabel('Sampling T')
    ax.set_ylabel(get_metric_label(metric))
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

    ax.set_title(f'Uncertainty in {get_metric_abbr(metric)} Across Models')
    ax.set_ylabel(get_metric_label(metric))
    ax.set_xlabel('')
    ax.set_xticks(sampling_names, labels)

    return ax

  def viusalise_slice_comparison(self, sample, slice_index, axis='sagittal'):
    fig, axes = plt.subplots(2, len(self.results_list) + 1, figsize=(5 * (len(self.results_list) + 1), 10))

    first_result = self.results_list[0]
    sample_results = first_result.for_sample(sample)
    structure_vol, original_dose_vol = sample_results.load_orig_volume()

    if axis == "sagittal":
      structure = structure_vol[:, :, :, slice_index]
      original_dose = original_dose_vol[:, :, slice_index]
    elif axis == "axial":
      structure = structure_vol[:, slice_index, :, :]
      original_dose = original_dose_vol[slice_index, :, :]
    elif axis == "coronal":
      structure = np.flip(structure_vol[:, :, slice_index, :], 1)
      original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV masks using selector
    ct_slice = first_result.dataset.selector.get_ct(structure)[0]
    ptv_doses = first_result.dataset.selector.get_component(structure, 'ptv_doses')

    axes[1, 0].imshow(ct_slice, cmap='gray')
    draw_ptv_overlay(axes[1, 0], ptv_doses)
    draw_ptv_contours(axes[1, 0], ptv_doses)
    axes[1, 0].set_title('CT Slice')
    axes[1, 0].axis('off')

    im1 = axes[0, 0].imshow(original_dose, cmap='gray', vmin=0, vmax=original_dose.max())
    draw_ptv_contours(axes[0, 0], ptv_doses)
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
      draw_ptv_contours(axes[0, i + 1], ptv_doses)
      axes[0, i + 1].set_title(f'{self.result_labels[i]} - Predicted Dose')
      axes[0, i + 1].axis('off')
      plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04)

      im_err = axes[1, i + 1].imshow(
        original_dose - pred_dose,
        cmap='bwr',
        vmin=-np.max(np.abs(original_dose - pred_dose)),
        vmax=np.max(np.abs(original_dose - pred_dose))
      )
      draw_ptv_contours(axes[1, i + 1], ptv_doses)
      axes[1, i + 1].set_title(f'{self.result_labels[i]} - Error (original - predicted)')
      axes[1, i + 1].axis('off')
      plt.colorbar(im_err, ax=axes[1, i + 1], fraction=0.046, pad=0.04)

    if fig is not None:
      fig.tight_layout()

    return fig, axes

  def visualise_slice_comparison_with_ct(self, sample, slice_index, axis='sagittal', dose_cmap='jet', dose_alpha=0.5, show_oar=True):
    """
    Visualize slice comparison with CT underneath transparent dose distributions.

    Args:
        sample: Sample name to visualize
        slice_index: Index of the slice to visualize
        axis: 'sagittal', 'axial', or 'coronal'
        dose_cmap: Colormap for dose overlay (default: 'jet')
        dose_alpha: Transparency of dose overlay (default: 0.5)
        show_oar: Whether to show OAR contours (default: True)

    Returns:
        Tuple of (figure, axes)
    """
    fig, axes = plt.subplots(2, len(self.results_list) + 1, figsize=(5 * (len(self.results_list) + 1), 10))

    first_result = self.results_list[0]
    sample_results = first_result.for_sample(sample)
    structure_vol, original_dose_vol = sample_results.load_orig_volume()

    if axis == "sagittal":
      structure = structure_vol[:, :, :, slice_index]
      original_dose = original_dose_vol[:, :, slice_index]
      structure = np.flip(structure, [1, 2])
      original_dose = np.flip(original_dose)
    elif axis == "axial":
      structure = structure_vol[:, slice_index, :, :]
      original_dose = original_dose_vol[slice_index, :, :]
    elif axis == "coronal":
      structure = np.flip(structure_vol[:, :, slice_index, :], 1)
      original_dose = np.flip(original_dose_vol[:, slice_index, :], 0)
    else:
      raise ValueError(f'Invalid axis: {axis}. Choose from "axial", "sagittal", or "coronal".')

    # Get CT and PTV masks using selector
    ct_slice = first_result.dataset.selector.get_ct(structure)[0]
    ptv_doses = first_result.dataset.selector.get_component(structure, 'ptv_doses')

    # Get OAR masks if needed
    oar_masks = None
    if show_oar:
      oar_masks = first_result.dataset.selector.get_component(structure, 'oar_v3')

    dose_max = original_dose.max()

    # Row 0, Col 0: CT with original dose overlay
    axes[0, 0].imshow(ct_slice, cmap='gray')
    dose_masked = np.ma.masked_where(original_dose < 0.01 * dose_max, original_dose)
    im1 = axes[0, 0].imshow(dose_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=0, vmax=dose_max)
    draw_ptv_contours(axes[0, 0], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(axes[0, 0], oar_masks)
    axes[0, 0].set_title('Original Dose on CT')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04, label='Dose (Gy)')

    # Row 1, Col 0: CT with PTV only (reference)
    axes[1, 0].imshow(ct_slice, cmap='gray')
    draw_ptv_overlay(axes[1, 0], ptv_doses)
    draw_ptv_contours(axes[1, 0], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(axes[1, 0], oar_masks)
    axes[1, 0].set_title('CT with PTV' + (' & OAR' if show_oar else ''))
    axes[1, 0].axis('off')

    for i, result in enumerate(self.results_list):
      sample_results = result.for_sample(sample)
      pred_dose_vol = sample_results.load_pred_volume()
      if axis == "sagittal":
        pred_dose = pred_dose_vol[:, :, slice_index]
        pred_dose = np.flip(pred_dose)
      elif axis == "axial":
        pred_dose = pred_dose_vol[slice_index, :, :]
      else:  # coronal
        pred_dose = np.flip(pred_dose_vol[:, slice_index, :], 0)

      # Row 0: CT with predicted dose overlay
      axes[0, i + 1].imshow(ct_slice, cmap='gray')
      pred_masked = np.ma.masked_where(pred_dose < 0.01 * dose_max, pred_dose)
      im = axes[0, i + 1].imshow(pred_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=0, vmax=dose_max)
      draw_ptv_contours(axes[0, i + 1], ptv_doses)
      if show_oar and oar_masks is not None:
        draw_oar_contours(axes[0, i + 1], oar_masks)
      axes[0, i + 1].set_title(f'{self.result_labels[i]} - Predicted')
      axes[0, i + 1].axis('off')
      plt.colorbar(im, ax=axes[0, i + 1], fraction=0.046, pad=0.04, label='Dose (Gy)')

      # Row 1: CT with error overlay
      error = original_dose - pred_dose
      error_max = np.max(np.abs(error))
      axes[1, i + 1].imshow(ct_slice, cmap='gray')
      error_masked = np.ma.masked_where(np.abs(error) < 0.01 * error_max, error)
      im_err = axes[1, i + 1].imshow(error_masked, cmap='bwr', alpha=dose_alpha, vmin=-error_max, vmax=error_max)
      draw_ptv_contours(axes[1, i + 1], ptv_doses)
      if show_oar and oar_masks is not None:
        draw_oar_contours(axes[1, i + 1], oar_masks)
      axes[1, i + 1].set_title(f'{self.result_labels[i]} - Error')
      axes[1, i + 1].axis('off')
      plt.colorbar(im_err, ax=axes[1, i + 1], fraction=0.046, pad=0.04, label='Error (Gy)')

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
