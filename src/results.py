import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Optional
from itertools import cycle
from scipy.stats import rankdata
from matplotlib.colors import to_rgba, to_rgb, BoundaryNorm
from matplotlib.patches import Patch, Polygon
import matplotlib.gridspec as gridspec
import matplotlib.transforms as transforms
from pathlib import Path

import json

from data import GDPHMM_OAR, GDPHMM_DOSE_MAX, GDPHMM_OAR_V3, GDPHMM_DATA_BASE_PATH
from evaluator import GDPHMM_UNIQUE_OAR

def rename_organs(series):
  series = series.str.replace('-PTV', '')
  series = series.str.replace('_05', '')
  series = series.str.replace('_03', '')
  series = series.str.replace('_', ' ')
  series = series.str.replace('SpinalCord', 'Spinal Cord')
  series = series.str.replace('BrainStem', 'Brain Stem')
  series = series.str.replace('ParotidCon', 'Parotid Contralateral')
  series = series.str.replace('ParotidIps', 'Parotid Ipsilateral')
  series = series.str.replace('OCavity', 'Oral Cavity')
  series = series.str.replace('PharConst', 'Pharynx Constrictor')
  series = series.str.replace('Cochlea L', 'Cochlea Left')
  series = series.str.replace('Cochlea R', 'Cochlea Right')
  series = series.str.replace('OpticNerve L', 'Optic Nerve Left')
  series = series.str.replace('OpticNerve R', 'Optic Nerve Right')
  series = series.str.replace('Submand', 'Submandibular')
  series = series.str.replace('SubmandibularR', 'Submandibular Right')
  series = series.str.replace('SubmandibularL', 'Submandibular Left')
  series = series.str.replace('Lens L', 'Lens Left')
  series = series.str.replace('Lens R', 'Lens Right')
  series = series.str.replace('Chiasm', 'Chiasma opticum')
  return series
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
  'rmse': 'Root Mean Squared Error',
  'psnr': 'Peak Signal-to-Noise Ratio (PSNR)',
  'ssim': 'Structural Similarity Index Measure (SSIM)',
  'ms_ssim': 'Multi-Scale Structural Similarity Index Measure (MS-SSIM)',
  'dvh': 'DVH Score',
  'dose_score': 'Dose Score (MAE)',
  'homogeneity_index': 'Homogeneity Index',
  'mse_body': 'Mean Squared Error (MSE) Body',
  'rmse_body': 'Root Mean Squared Error',
  'psnr_body': 'Peak Signal-to-Noise Ratio (PSNR) Body',
  'dose_score_body': 'Dose Score (MAE) Body',
  'var_laplacian_diff': 'Sharpness Metric (Variance of Laplacian Difference)',
  'mae_challenge': 'Mean Absolute Error (MAE) Challenge',
  'agreement/agreement_all': 'OAR Constraint Agreement',
  'disagreement/disagreement_all': 'OAR Constraint Disagreement',
  'agreement/agreement_all_percentage': 'OAR Constraint Agreement',
  'disagreement/disagreement_all_percentage': 'OAR Constraint Disagreement',
  'gamma_3_3/pass_rate': 'Gamma Index Pass Rate',
  'gamma_2_2/pass_rate': 'Gamma Index 2%/2mm Pass Rate',
  'gamma_5_5/pass_rate': 'Gamma Index 5%/5mm Pass Rate',
  'dvh_auc/OAR/diff': 'OAR DVH Area Between Curves',
  'mean_dose/high/abs_diff': 'PTV Dose Distance',
  'homogeneity_index/high/abs_diff': 'Homogeneity Index Distance',
  'paddick_conformity_index/high/abs_diff': 'Paddick Conformity Index Distance',
  'gradient_index/high/abs_diff': 'Paddick Gradient Index Distance',
}

METRIC_ABBRS = {
  'mse': 'MSE',
  'rmse': 'RMSE',
  'psnr': 'PSNR',
  'ssim': 'SSIM',
  'ms_ssim': 'MS-SSIM',
  'dvh': 'DVH Score',
  'dose_score': 'Dose Score (MAE)',
  'homogeneity_index': 'HI',
  'mse_body': 'MSE Body',
  'rmse_body': 'RMSE',
  'psnr_body': 'PSNR Body',
  'dose_score_body': 'Dose Score (MAE) Body',
  'var_laplacian_diff': 'Sharpness',
  'mae_challenge': 'MAE Challenge',
  'agreement/agreement_all': 'Agreement',
  'disagreement/disagreement_all': 'Disagreement',
  'agreement/agreement_all_percentage': 'Agreement',
  'disagreement/disagreement_all_percentage': 'Disagreement',
  'gamma_5_5/pass_rate': 'Gamma\n5%/5mm',
  'gamma_3_3/pass_rate': 'Gamma\nIndex',
  'gamma_2_2/pass_rate': 'Gamma\n2%/2mm',
  'gamma_1_1/pass_rate': 'Gamma\n1%/1mm',
  'dvh_auc/OAR/diff': 'OAR DVH\nABC',
  'mean_dose/high/abs_diff': 'PTV Dose\nDistance',
  'homogeneity_index/high/abs_diff': 'HI\nDistance',
  'paddick_conformity_index/high/abs_diff': 'Paddick CI\nDistance',
  'gradient_index/high/abs_diff': 'Paddick GI\nDistance',
}

METRIC_UNITS = {
  'mse': None,
  'rmse': 'Gy',
  'psnr': None,
  'ssim': None,
  'ms_ssim': None,
  'dvh': 'Gy',
  'dose_score': 'Gy',
  'homogeneity_index': None,
  'mse_body': None,
  'rmse_body': 'Gy',
  'psnr_body': None,
  'dose_score_body': 'Gy',
  'var_laplacian_diff': None,
  'mae_challenge': None,
  'agreement/agreement_all': None,
  'disagreement/disagreement_all': None,
  'agreement/agreement_all_percentage': '%',
  'disagreement/disagreement_all_percentage': '%',
  'gamma_5_5/pass_rate': '%',
  'gamma_3_3/pass_rate': '%',
  'gamma_2_2/pass_rate': '%',
  'gamma_1_1/pass_rate': '%',
  'dvh_auc/OAR/diff': 'Gy',
  'mean_dose/high/abs_diff': 'Gy',
  'homogeneity_index/high/abs_diff': None,
  'paddick_conformity_index/high/abs_diff': None,
  'gradient_index/high/abs_diff': None,
}

METRICS_DEFAULT = [
  'rmse', 'dose_score', 'dvh',
  'var_laplacian_diff', 'dose_score_body', 'homogeneity_index',
  'ssim', 'mae_challenge', 'psnr_body'
]

METRICS_PRED = [
  'vol_disagreement/disagreement_all_percentage', 'vol_dvh_auc/OAR/diff', 'vol_dvh',
  'vol_mean_dose/high/abs_diff', 'vol_paddick_conformity_index/high/abs_diff', 'vol_gradient_index/high/abs_diff'
]

METRICS_PLAN = [
  'vol_disagreement/disagreement_all_percentage', 'vol_dvh_auc/OAR/diff', 'vol_rmse_body', 'vol_gamma_3_3/pass_rate',
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
  'agreement/agreement_all',
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

def get_metric_abbr(metric_name: str, with_lb=False, with_unit=True) -> str:
  """
  Get a human-readable abbreviation for a given metric name.

  Args:
    metric_name: Name of the metric (can be simple like 'rmse' or
                  compound like 'vol_agreement/agreement_all')
    with_lb: If True, use line breaks between name and unit.
    with_unit: If True, append the unit (e.g. [Gy]) to the abbreviation.

  Returns:
    A human-readable abbreviation for the metric.
  """
  if metric_name.startswith('vol_'):
    metric_name = metric_name[4:]  # Remove 'vol_' prefix

  matched_key = None
  if metric_name in METRIC_ABBRS:
    matched_key = metric_name
  else:
    for key in METRIC_ABBRS.keys():
      if key.startswith(metric_name):
        matched_key = key
        break

  if matched_key is not None:
    abbr = METRIC_ABBRS[matched_key]
    unit = METRIC_UNITS.get(matched_key) if with_unit else None
  else:
    abbr = metric_name
    unit = None

  if unit:
    if with_lb and '\n' in abbr:
      abbr = f'{abbr} [{unit}]'
    else:
      sep = '\n' if with_lb else ' '
      abbr = f'{abbr}{sep}[{unit}]'

  if not with_lb:
    abbr = abbr.replace('\n', ' ')

  return abbr

def get_metric_unit(metric_name: str) -> str | None:
  """
  Get the unit for a given metric name.

  Args:
    metric_name: Name of the metric (can be simple like 'rmse' or
                  compound like 'vol_agreement/agreement_all')

  Returns:
    The unit string (e.g. 'Gy', '%') or None if the metric has no unit.
  """
  if metric_name.startswith('vol_'):
    metric_name = metric_name[4:]

  if metric_name in METRIC_UNITS:
    return METRIC_UNITS[metric_name]

  for key in METRIC_UNITS.keys():
    if key.startswith(metric_name):
      return METRIC_UNITS[key]

  return None

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

def draw_beam_overlay(ax, beam_plate, color='yellow', alpha=0.3):
  """
  Draw filled overlay for the beam plate on the given axes.

  Args:
      ax: Matplotlib axes
      beam_plate: Array of shape (1, H, W) or (H, W) with beam plate values
      color: Color for the overlay
      alpha: Maximum transparency of the overlay
  """
  if beam_plate.ndim == 3:
    beam_plate = beam_plate[0]
  beam_mask = beam_plate > 0
  if not beam_mask.any():
    return
  overlay = np.zeros((*beam_plate.shape, 4))
  rgb = to_rgb(color)
  overlay[beam_mask, 0] = rgb[0]
  overlay[beam_mask, 1] = rgb[1]
  overlay[beam_mask, 2] = rgb[2]
  # Scale alpha by beam intensity for a nicer look
  max_val = beam_plate.max()
  if max_val > 0:
    overlay[beam_mask, 3] = alpha * beam_plate[beam_mask] / max_val
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
      'body',
      'oar_v3',
      'ptv_doses',
      'beam_plate',
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

def setup_fig_axes(metrics, sharey=False):
  if len(metrics) % 4 == 0:
    n_cols = 4
    n_rows = len(metrics) // 4
  else:
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

  fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), squeeze=False, sharey=sharey)
  axes = axes.flatten()
  return fig, axes


class BaseResults:
  def __init__(
    self,
    model_name,
    sampling_name,
    epoch,
    dataset_split,
    dataset,
    distillation_name=None,
    distillation_round=None,
  ):
    self.model_name = model_name
    self.sampling_name = sampling_name
    self.epoch = epoch
    self.dataset_split = dataset_split
    self.distillation_name = distillation_name
    self.distillation_round = distillation_round

    self.base_path = SAMPLES_PATH / model_name
    if distillation_name and distillation_round:
      self.base_path = self.base_path / f'distill-{distillation_name}' / f'round-{distillation_round}'
    self.base_path = self.base_path / sampling_name / f'{dataset_split}_{epoch}'

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

  def plot_quality_over_slice_index_single(self, metric, ax=None, relative_to_center=False, agg='mean'):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # Plot one line per result in results_list
    for result, label in zip(self.results_list, self.result_labels):
      if relative_to_center:
        df = result.slice_metrics[['file_name', 'slice_index', metric]].copy()
        mid = df.groupby('file_name')['slice_index'].transform(lambda s: round((s.min() + s.max()) / 2))
        df['rel_index'] = df['slice_index'] - mid
        grouped = df.groupby('rel_index')[metric]
      else:
        grouped = result.slice_metrics.groupby('slice_index')[metric]

      if agg == 'mean':
        center = grouped.mean()
        err_lo = grouped.std()
        err_hi = err_lo
      elif agg == 'median':
        center = grouped.median()
        err_lo = center - grouped.quantile(0.25)
        err_hi = grouped.quantile(0.75) - center

      ax.errorbar(center.index, center.values, yerr=[err_lo.values, err_hi.values], fmt='o-', capsize=3, label=label, alpha=0.7)

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    index_label = 'Relative Slice Index (0 = center)' if relative_to_center else 'Slice Index'
    agg_label = 'Mean ± SD' if agg == 'mean' else 'Median ± IQR'
    ax.set_title(f'{get_metric_abbr(metric, with_unit=False)} vs {index_label} \n{dir_symbol} is better')
    ax.set_xlabel(index_label)
    ax.set_ylabel(agg_label + ' ' + get_metric_abbr(metric))

    # Only show legend if there are multiple results
    if len(self.results_list) > 1:
      ax.legend()

    return ax

  def plot_quality_over_slice_index(self, metrics=METRICS_DEFAULT, relative_to_center=False, agg='mean'):
    if isinstance(metrics, str):
      return self.plot_quality_over_slice_index_single(metrics, relative_to_center=relative_to_center, agg=agg)

    fig, axes = setup_fig_axes(metrics, sharey=False)

    for i, metric in enumerate(metrics):
      metric = metric.replace('vol_', '')  # Remove 'vol_' prefix for plotting
      self.plot_quality_over_slice_index_single(metric, ax=axes[i], relative_to_center=relative_to_center, agg=agg)

    index_label = 'Relative Slice Index (0 = center)' if relative_to_center else 'Slice Index'
    fig.suptitle(f'Quality vs {index_label}', fontsize=16)
    fig.tight_layout()

    if len(self.results_list) > 1:
      handles, labels = axes[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper right', title='', ncol=8)

      for ax in axes:
        legend = ax.get_legend()
        if legend:
          legend.remove()

    return fig, axes

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
  def __init__(self, model_name, sampling_name, epoch, dataset_split, dataset, distillation_name=None, distillation_round=None):
    super().__init__(model_name, sampling_name, epoch, dataset_split, dataset, distillation_name, distillation_round)

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
    self.volume_metrics['vol_disagreement/disagreement_all'] = 1 - self.volume_metrics['vol_agreement/agreement_all']
    self.volume_metrics['vol_agreement/agreement_all_percentage'] = self.volume_metrics['vol_agreement/agreement_all'] * 100
    self.volume_metrics['vol_disagreement/disagreement_all_percentage'] = self.volume_metrics['vol_disagreement/disagreement_all'] * 100

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
      model_results.dataset,
      model_results.distillation_name,
      model_results.distillation_round
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

    item = self.dataset[original_idx]
    # Handle both regular datasets (3 values) and WarmStartCachedDataset (5 values)
    structure, dose = item[0], item[1]
    structure = structure.numpy()
    dose = self.dataset.denormalize_target(dose)[0].numpy()

    return structure, dose


  def load_pred_volume(self):
    return np.load(self.base_path / f'{self.sample}.npy')[0]


  def plot_volume_dvh(self, ax=None, highlight=None, fade_alpha=0.3):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    structure, original_dose = self.load_orig_volume()
    pred_dose = self.load_pred_volume()

    structure = structure[np.newaxis, ...]  # Add batch dimension
    ptv_doses_channels = self.dataset.selector.get_ptv_doses(structure)
    ptv_masks = ptv_doses_channels > 0
    oar_masks = self.dataset.selector.get_component(structure, 'oar_v3') > 0

    # Build PTV level names from PTV_DICT.json
    with open(GDPHMM_DATA_BASE_PATH / 'gdphmm-hn' / 'incoming' / 'PTV_DICT.json') as f:
      ptv_info = json.load(f)[self.sample.split('+')[0]]

    # Channels sorted by dose descending: High, Mid, Low
    ptv_level_order = ['PTV_High', 'PTV_Mid', 'PTV_Low']
    ptv_names = []
    for i, level in enumerate(ptv_level_order):
      if ptv_masks[0, i].sum() > 0 and level in ptv_info:
        ptv_names.append(f'{level.replace("PTV_", "PTV ")} ({ptv_info[level]["PDose"]:.0f} Gy)')

    ptv_masks_present = ptv_masks[0, :len(ptv_names)]
    all_oar_names = rename_organs(pd.Series(GDPHMM_OAR_V3)).tolist()

    # Filter to unique OARs present in the volume
    oar_present = [(name, raw_name, mask) for name, raw_name, mask in zip(all_oar_names, GDPHMM_OAR_V3, oar_masks[0]) if raw_name in GDPHMM_UNIQUE_OAR and mask.sum() > 0]
    oar_present.sort(key=lambda x: x[0])
    oar_names_present = [name for name, _, _ in oar_present]
    oar_masks_present = np.stack([mask for _, _, mask in oar_present]) if oar_present else np.empty((0, *ptv_masks_present.shape[1:]))

    all_masks = np.concatenate([ptv_masks_present, oar_masks_present], axis=0)
    all_names = ptv_names + oar_names_present

    bins = np.linspace(0, np.max(original_dose), 200)
    ptv_colors = ['#D62728', '#E377C2', '#FF7F0E']
    oar_cmap = plt.cm.tab20
    oar_colors = [oar_cmap(i) for i in range(len(oar_names_present))]
    all_colors = ptv_colors[:len(ptv_names)] + oar_colors
    handles, labels = [], []
    for name, mask, color in zip(all_names, all_masks, all_colors):
      x, y_orig = compute_dvh(original_dose, mask, bins)
      _, y_pred = compute_dvh(pred_dose, mask, bins)
      alpha = fade_alpha if highlight and name not in highlight else 1.0
      lw = 1.0 if highlight and name not in highlight else 1.5
      ax.plot(x, y_orig, color=color, linestyle='-', alpha=alpha, lw=lw)
      ax.plot(x, y_pred, color=color, linestyle='--', alpha=alpha, lw=lw)
      handles.append(plt.Line2D([0], [0], color=color, lw=2, alpha=alpha))
      labels.append(name)

    ax.set_xlabel('Dose [Gy]')
    ax.set_ylabel('Volume Fraction ≥ Dose')
    ax.set_title('DVH Comparison')
    ax.set_xlim(0, np.max(original_dose) + 1)
    ax.set_ylim(0, 1.02)
    ax.grid(True, linestyle=':')
    ax.legend(handles, labels, fontsize=8, title='', loc='center left', bbox_to_anchor=(1.0, 0.5))

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
      dataset=first_result.dataset,
      distillation_name=first_result.distillation_name,
      distillation_round=first_result.distillation_round
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
      groups_to_add = ['']  # Default group
    elif isinstance(other, ModelComparison):
      to_add = other.results_list
      labels_to_add = other.result_labels
      groups_to_add = other.result_groups
    elif isinstance(other, (list, tuple)):
      to_add = list(other)
      labels_to_add = [f'{r.model_name} ({r.sampling_name})' for r in to_add]
      groups_to_add = [''] * len(to_add)  # Default groups
    else:
      raise TypeError(f'Cannot add {type(other).__name__} to ModelComparison')

    new_results = list(self.results_list)
    new_labels = list(self.result_labels)
    new_groups = list(self.result_groups)
    for result, label, group in zip(to_add, labels_to_add, groups_to_add):
      if result not in new_results:
        new_results.append(result)
        new_labels.append(label)
        new_groups.append(group)
    return ModelComparison(new_results, new_labels, new_groups)

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
    new_groups = []
    for result, label, group in zip(self.results_list, self.result_labels, self.result_groups):
      if result not in to_exclude:
        new_results.append(result)
        new_labels.append(label)
        new_groups.append(group)

    if len(new_results) == 0:
      raise ValueError('Cannot create ModelComparison with no results')

    return ModelComparison(new_results, new_labels, new_groups)

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

  def plot_metric_distribution(self, metric, ax=None, sort=False, outliers=True, skip_colors=0, colors=None, show_is_better=True, highlight=None, muted_alpha=0.5):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    if isinstance(highlight, str):
      highlight = [highlight]

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

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    unique_groups = df['group'].unique()
    if colors is not None and isinstance(colors, dict):
      palette = {g: colors.get(g, prop_cycle[(i + skip_colors) % len(prop_cycle)]) for i, g in enumerate(unique_groups)}
    elif colors is not None:
      palette = {g: colors[i % len(colors)] for i, g in enumerate(unique_groups)}
    elif skip_colors > 0:
      palette = {g: prop_cycle[(i + skip_colors) % len(prop_cycle)] for i, g in enumerate(unique_groups)}
    else:
      palette = None

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
      sns.boxplot(x='value', y='model', data=df, ax=ax, order=order, hue='group', dodge=False, medianprops=medianprops, palette=palette)
    else:
      sns.boxplot(x='value', y='model', data=df, ax=ax, order=order, hue='group', dodge=False, whis=1.5, medianprops=medianprops, palette=palette)
      vmin, vmax, lim_min, lim_max = iqr_limits_from_df(df, 'model', 'value', whis=1.5)
      ax.set_xlim(lim_min, lim_max)

      model_order = order if order is not None else df['model'].unique().tolist()
      model_to_y = {model: i for i, model in enumerate(model_order)}

      above_mask = df['value'] > vmax
      below_mask = df['value'] < vmin
      above_y = df.loc[above_mask, 'model'].map(model_to_y)
      below_y = df.loc[below_mask, 'model'].map(model_to_y)
      ylim = ax.get_ylim()
      ax.plot(np.full(len(above_y), vmax), above_y, marker=">", linestyle="none", color="black")
      ax.plot(np.full(len(below_y), vmin), below_y, marker="<", linestyle="none", color="black")
      ax.set_ylim(ylim)

    if highlight is not None:
      from matplotlib.patches import PathPatch
      model_order = order if order is not None else df['model'].unique().tolist()
      box_patches = sorted(
        [p for p in ax.patches if isinstance(p, PathPatch)],
        key=lambda p: (p.get_path().get_extents().y0 + p.get_path().get_extents().y1) / 2,
      )
      for i, patch in enumerate(box_patches):
        if i < len(model_order) and model_order[i] not in highlight:
          patch.set_alpha(muted_alpha)
      for tick, label_text in zip(ax.get_yticklabels(), model_order):
        if label_text not in highlight:
          tick.set_alpha(muted_alpha)

    dir_symbol = '→' if is_higher_better(metric) else '←'
    ax.set_title(get_metric_label(metric))
    ax.set_xlabel(get_metric_abbr(metric))
    ax.set_ylabel('')

    if show_is_better:
      blend = transforms.blended_transform_factory(ax.transData, ax.transAxes)
      offset = transforms.ScaledTranslation(-60/72, -7/72, ax.figure.dpi_scale_trans)
      ax.text(ax.get_xlim()[0], 0, f'{dir_symbol} is better', transform=blend + offset,
              ha='left', va='top', fontsize='medium', color='gray')

    if self.has_groups:
      ax.legend(title='')

    return ax

  def plot_metric_distributions(self, metrics=METRICS_DEFAULT, sort=False, outliers=True, skip_colors=0, colors=None, highlight=None):
    if outliers == True:
      outliers = metrics
    elif outliers == False:
      outliers = []

    fig, axes = setup_fig_axes(metrics, sharey=False)

    for i, metric in enumerate(metrics):
      self.plot_metric_distribution(metric, ax=axes[i], sort=sort, outliers=metric in outliers, skip_colors=skip_colors, colors=colors, highlight=highlight)

    fig.suptitle('Metric Distributions Across Models', fontsize=16)
    fig.tight_layout()

    if self.has_groups:
      handles, labels = axes[0].get_legend_handles_labels()
      fig.legend(handles, labels, loc='upper right', title='', ncol=8)

      for ax in axes:
        ax.get_legend().remove()

    return fig, axes

  def plot_metric_scatter(self, metric, label_x, label_y, better_label=None, jitter=False, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(4, 4))

    vm_x = self.volume_metrics[self.volume_metrics['label'] == label_x].set_index('file_name')
    vm_y = self.volume_metrics[self.volume_metrics['label'] == label_y].set_index('file_name')

    x = vm_x[metric]
    y = vm_y.reindex(x.index)[metric]
    mask = ~(x.isna() | y.isna())
    x, y = x[mask], y[mask]

    if jitter:
      rng = np.random.default_rng(42)
      spread = (x.max() - x.min()) * 0.05
      x = x + rng.normal(0, spread, size=len(x))
      y = y + rng.normal(0, spread, size=len(y))

    all_vals = pd.concat([x, y])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    lo, hi = lo - margin, hi + margin

    # Determine which triangle (above/below diagonal) is "y better"
    higher_better = is_higher_better(metric)
    # Above diagonal: y > x. For higher_better metrics, y > x means y is better.
    # For lower_better metrics, y < x (below diagonal) means y is better.
    y_better_above = higher_better

    # If better_label is label_x, flip: green where x outperforms y
    if better_label == label_x:
      y_better_above = not y_better_above

    above_verts = [(lo, lo), (lo, hi), (hi, hi)]
    below_verts = [(lo, lo), (hi, hi), (hi, lo)]

    green_verts = above_verts if y_better_above else below_verts
    red_verts = below_verts if y_better_above else above_verts

    ax.add_patch(Polygon(green_verts, closed=True, facecolor='#2ca02c', alpha=0.08))
    ax.add_patch(Polygon(red_verts, closed=True, facecolor='#d62728', alpha=0.08))

    ax.plot([lo, hi], [lo, hi], 'k', linewidth=0.2, alpha=0.5)
    ax.scatter(x, y, s=20, alpha=0.6, c='0.3')

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    ax.set_yticks(ax.get_xticks())
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)

    unit = get_metric_unit(metric)
    unit_suffix = f' [{unit}]' if unit else ''
    ax.set_xlabel(f'{label_x}{unit_suffix}')
    ax.set_ylabel(f'{label_y}{unit_suffix}')
    ax.set_title(get_metric_abbr(metric, with_lb=False, with_unit=False))

    return ax

  def plot_metric_scatters(self, metrics, label_x, label_y, better_label=None, jitter=True):
    if jitter == True:
      jitter = metrics
    elif jitter == False:
      jitter = []

    fig, axes = setup_fig_axes(metrics)

    for i, metric in enumerate(metrics):
      self.plot_metric_scatter(metric, label_x, label_y, better_label=better_label, jitter=metric in jitter, ax=axes[i])

    for j in range(len(metrics), len(axes)):
      axes[j].set_visible(False)

    fig.suptitle(f'{label_x} vs {label_y} Comparison')
    fig.tight_layout()

    return fig, axes

  def plot_inference_times(self, entity='volume', unit='minutes', sort=True, ax=None):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    data = []
    labels = []
    groups = []
    for result, label, group in zip(self.results_list, self.result_labels, self.result_groups):
      if entity == 'volume':
        data.extend(result.volume_metrics['duration'])
        labels.extend([label] * len(result.volume_metrics['duration']))
        groups.extend([group] * len(result.volume_metrics['duration']))
      elif entity == 'slice':
        data.extend(result.slice_metrics['duration'])
        labels.extend([label] * len(result.slice_metrics['duration']))
        groups.extend([group] * len(result.slice_metrics['duration']))
      else:
        raise ValueError(f"Unknown entity: {entity}")

    df = pd.DataFrame({'model': labels, 'inference_time': data, 'group': groups})
    if unit == 'seconds':
      short_unit = 's'
      pass  # already in seconds
    elif unit == 'milliseconds':
      short_unit = 'ms'
      df['inference_time'] = df['inference_time'] * 1000.0
    elif unit == 'minutes':
      short_unit = 'min'
      df['inference_time'] = df['inference_time'] / 60.0
    else:
      raise ValueError(f"Unknown unit: {unit}")

    order = df.groupby('model')['inference_time'].median().sort_values().index.tolist() if sort else None
    sns.boxplot(x='inference_time', y='model', data=df, ax=ax, hue='group', dodge=False, order=order)
    ax.set_title('Inference Time Distribution per ' + ('Volume' if entity == 'volume' else 'Slice'))
    ax.set_xlabel(f'Inference Time [{short_unit}]')
    ax.set_ylabel('')

    if self.has_groups:
      ax.legend(title='')

    return ax

  def plot_ptv_evaluation(self, ax=None, metrics=None):
    def plot_mean_dose(ax):
      ax.plot([55, 75], [55, 75], color='0.8', linestyle='-', linewidth=0.8)
      sns.scatterplot(self.volume_metrics, x=f'vol_mean_dose/high/target', y=f'vol_mean_dose/high/pred', hue='label', ax=ax, legend='full')
      ax.set_title('PTV Mean Dose')
      ax.set_xlabel('Target Mean Dose [Gy]')
      ax.set_ylabel('Predicted Mean Dose [Gy]')
      ax.xaxis.get_major_locator().set_params(integer=True)
      ax.yaxis.get_major_locator().set_params(integer=True)

    def plot_homogeneity_index(ax):
      ax.fill_between([0, 0.4], [0, 0.4], 0.4, color='red', alpha=0.1)
      ax.fill_between([0, 0.4], [0, 0.4], 0, color='green', alpha=0.1)
      sns.scatterplot(self.volume_metrics, x=f'vol_homogeneity_index/high/target', y=f'vol_homogeneity_index/high/pred', hue='label', ax=ax, legend='full')
      ax.set_title(f'Homogeneity Index')
      ax.set_xlabel('Target Homogeneity Index')
      ax.set_ylabel('Predicted Homogeneity Index')

    def plot_paddick_conformity_index(ax):
      ax.fill_between([0, 1], [0, 1], 1, color='green', alpha=0.1)
      ax.fill_between([0, 1], [0, 1], 0, color='red', alpha=0.1)
      sns.scatterplot(self.volume_metrics, x=f'vol_paddick_conformity_index/high/target', y=f'vol_paddick_conformity_index/high/pred', hue='label', ax=ax, legend='full')
      ax.set_title(f'Paddick Conformity Index')
      ax.set_xlabel('Target Conformity Index')
      ax.set_ylabel('Predicted Conformity Index')

    def plot_gradient_index(ax):
      ax.fill_between([0, 100], [0, 100], 100, color='red', alpha=0.1)
      ax.fill_between([0, 100], [0, 100], 0, color='green', alpha=0.1)
      sns.scatterplot(self.volume_metrics, x=f'vol_gradient_index/high/target', y=f'vol_gradient_index/high/pred', hue='label', ax=ax, legend='full')
      ax.set_title(f'Gradient Index')
      ax.set_xlabel('Target Gradient Index')
      ax.set_ylabel('Predicted Gradient Index')

    plotters = {
      'mean_dose': plot_mean_dose,
      'homogeneity_index': plot_homogeneity_index,
      'paddick_conformity_index': plot_paddick_conformity_index,
      'gradient_index': plot_gradient_index,
    }

    if metrics is None:
      metrics = list(plotters.keys())

    fig = None
    if ax is None:
      fig, axes = plt.subplots(1, len(metrics), figsize=(15 * len(metrics) / 4, 4))
      if len(metrics) == 1:
        axes = [axes]

    for a, metric in zip(axes, metrics):
      plotters[metric](a)

    handles, labels = axes[0].get_legend_handles_labels()
    if fig is not None:
      fig.legend(handles, labels, loc='lower center', title='', ncol=8, bbox_to_anchor=(0.5, -0.1))

    for a in axes:
      a.grid(True, color='lightgray', linestyle='--', linewidth=0.5)
      a.set_aspect('equal')
      a.get_legend().remove()
      a.set_facecolor('white')

    if fig is not None:
      fig.suptitle('PTV Evaluation')
      fig.tight_layout()

    return fig, axes

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

  def plot_quality_over_T(
      self,
      metrics,
      agg='mean',
      xscale='log',
      baseline=None,
      skip_colors=0,
      colors=None,
      show_uncertainty=True,
  ):
    fig, axes = setup_fig_axes(metrics)

    for metric, ax in zip(metrics, axes):
      self.plot_quality_over_T_single(
        metric,
        agg=agg,
        xscale=xscale,
        baseline=baseline,
        ax=ax,
        skip_colors=skip_colors,
        colors=colors,
        show_uncertainty=show_uncertainty,
      )
      ax.get_legend().remove()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', title='', ncol=8)
    fig.suptitle('Metrics vs Sampling T Across Models', fontsize=16)
    fig.tight_layout()

    return fig, axes


  def _collect_quality_over_T_data(self, metric, agg='mean'):
    """Collect aggregated data points for each model label."""
    data = {}
    for result, result_label in zip(self.results_list, self.result_labels):
      df = result.volume_metrics
      T = int(result.sampling_name.split('-')[-1])
      if agg == 'mean':
        y = df[metric].mean()
        y_lo = y - df[metric].std()
        y_hi = y + df[metric].std()
      elif agg == 'median':
        y = df[metric].median()
        y_lo = df[metric].quantile(0.25)
        y_hi = df[metric].quantile(0.75)
      label = result_label.replace('\n', ' ').replace(f' T=S={T}', '').replace(f' S={T}', '')
      data[label] = data.get(label, []) + [(T, y, y_lo, y_hi)]

    for label in data:
      data[label].sort(key=lambda p: p[0])
    return data


  def plot_quality_over_T_paired(
      self,
      metrics,
      agg='mean',
      xscale='log',
      baseline=None,
      skip_colors=0,
      ratio=True,
      colors=None,
      diff_color='black',
      show_uncertainty=True,
      background=None,
      bg_colors=None,
      bg_alpha=0.3,
      n_col_legend=None,
  ):
    """Plot each metric as a pair of panels: absolute values on top, ratio (or difference) below.

    Metrics are arranged in a grid with n_cols columns. Each metric occupies two rows
    (absolute values + ratio/difference), so the total grid has 2*n_metric_rows physical rows.
    """

    n_metrics = len(metrics)
    if n_metrics % 4 == 0:
      n_cols = 4
      n_metric_rows = n_metrics // 4
    else:
      n_cols = min(3, n_metrics)
      n_metric_rows = (n_metrics + n_cols - 1) // n_cols

    fig = plt.figure(figsize=(4.5 * n_cols, 4 * n_metric_rows))
    outer_gs = gridspec.GridSpec(n_metric_rows, n_cols, figure=fig, hspace=0.3, wspace=0.17, top=0.92)

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
    all_axes = []

    for idx, metric in enumerate(metrics):
      row = idx // n_cols
      col = idx % n_cols
      inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[2, 1], hspace=0.05)
      ax_top = fig.add_subplot(inner_gs[0])
      ax_bot = fig.add_subplot(inner_gs[1], sharex=ax_top)
      all_axes.append((ax_top, ax_bot))

      data = self._collect_quality_over_T_data(metric, agg=agg)
      labels = list(data.keys())

      # resolve colors per label
      label_colors = {}
      for i, label in enumerate(labels):
        if colors is not None and isinstance(colors, dict):
          label_colors[label] = colors.get(label, prop_cycle[(i + skip_colors) % len(prop_cycle)])
        elif colors is not None:
          label_colors[label] = colors[i % len(colors)]
        else:
          label_colors[label] = prop_cycle[(i + skip_colors) % len(prop_cycle)]

      # top panel: absolute values (plot first to establish autoscale limits)
      for label, points in data.items():
        c = label_colors[label]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        lo = [p[2] for p in points]
        hi = [p[3] for p in points]
        ax_top.plot(xs, ys, marker='o', label=label, color=c)
        if show_uncertainty:
          ax_top.fill_between(xs, lo, hi, alpha=0.2, color=c)

      if baseline is not None:
        y_baseline = baseline.volume_metrics[metric].mean() if agg == 'mean' else baseline.volume_metrics[metric].median()
        ax_top.axhline(y_baseline, color='black', linestyle='--', label='Baseline')

      ax_top.set_xscale(xscale)

      # background lines (added after foreground so they don't affect limits)
      if background is not None:
        xlim = ax_top.get_xlim()
        ylim_top = ax_top.get_ylim()
        ax_top.set_autoscale_on(False)
        bg_list = background if isinstance(background, list) else [background]
        bg_colors_dict = bg_colors or {}
        for bg_comp in bg_list:
          bg_data = bg_comp._collect_quality_over_T_data(metric, agg=agg)
          for bg_label, bg_points in bg_data.items():
            bg_c = bg_colors_dict.get(bg_label, 'gray')
            bg_xs = [p[0] for p in bg_points]
            bg_ys = [p[1] for p in bg_points]
            ax_top.plot(bg_xs, bg_ys, marker='o', color=bg_c, alpha=bg_alpha, label=bg_label, zorder=1)
        ax_top.set_xlim(xlim)
        ax_top.set_ylim(ylim_top)

      ax_top.set_title(get_metric_label(metric))
      ticks = sorted({p[0] for points in data.values() for p in points})
      ax_top.tick_params(labelbottom=False)

      if show_uncertainty:
        ax_top.set_ylabel(f'{agg.capitalize()}/{"std" if agg == "mean" else "IQR"}')
      else:
        ax_top.set_ylabel(agg.capitalize())

      dir_symbol = '↑' if is_higher_better(metric) else '↓'
      ax_top.text(0.95, 0.92, f'{dir_symbol} is better', transform=ax_top.transAxes,
                  ha='right', va='top', fontsize='medium', color='0.2')

      # bottom panel: ratio or difference
      if len(labels) == 2:
        label_a, label_b = labels
        pts_a = {p[0]: p for p in data[label_a]}
        pts_b = {p[0]: p for p in data[label_b]}
        common_xs = sorted(set(pts_a.keys()) & set(pts_b.keys()))

        higher_better = is_higher_better(metric)

        if ratio:
          vals = [pts_b[x][1] / pts_a[x][1] if pts_a[x][1] != 0 else float('nan') for x in common_xs]
          ax_bot.plot(common_xs, vals, marker='o', color=diff_color)
          ax_bot.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
          ax_bot.set_ylabel('Ratio')

          # shade regions and add corner labels
          # ratio = b/a: for lower-is-better, <1 means b better; for higher-is-better, >1 means b better
          ylim = ax_bot.get_ylim()
          margin = (ylim[1] - ylim[0]) * 0.3
          ylim = (ylim[0] - margin, ylim[1] + margin)
          better_below = label_b if not higher_better else label_a
          better_above = label_a if not higher_better else label_b
          ax_bot.axhspan(1.0, ylim[1], color=label_colors[better_above], alpha=0.08)
          ax_bot.axhspan(ylim[0], 1.0, color=label_colors[better_below], alpha=0.08)
          ax_bot.set_ylim(ylim)
          ax_bot.text(0.02, 0.02, f'{better_below} better', transform=ax_bot.transAxes,
                      ha='left', va='bottom', fontsize='x-small', color='0.2')
          ax_bot.text(0.02, 0.96, f'{better_above} better', transform=ax_bot.transAxes,
                      ha='left', va='top', fontsize='x-small', color='0.2')
        else:
          vals = [pts_b[x][1] - pts_a[x][1] for x in common_xs]
          ax_bot.plot(common_xs, vals, marker='o', color=diff_color)
          ax_bot.axhline(0.0, color='gray', linestyle='--', linewidth=0.8)

          ax_bot.set_ylabel(f'Difference ({label_b} − {label_a})')

          ylim = ax_bot.get_ylim()
          margin = (ylim[1] - ylim[0]) * 0.3
          ylim = (ylim[0] - margin, ylim[1] + margin)
          better_below = label_b if not higher_better else label_a
          better_above = label_a if not higher_better else label_b
          ax_bot.axhspan(0.0, ylim[1], color=label_colors[better_above], alpha=0.08)
          ax_bot.axhspan(ylim[0], 0.0, color=label_colors[better_below], alpha=0.08)
          ax_bot.set_ylim(ylim)
          ax_bot.text(0.02, 0.02, f'{better_below} better', transform=ax_bot.transAxes,
                      ha='left', va='bottom', fontsize='x-small', color='0.2')
          ax_bot.text(0.02, 0.96, f'{better_above} better', transform=ax_bot.transAxes,
                      ha='left', va='top', fontsize='x-small', color='0.2')
      else:
        # fallback for >2 models: show all pairwise ratios vs first model
        ref_label = labels[0]
        pts_ref = {p[0]: p for p in data[ref_label]}
        for i, label in enumerate(labels[1:], 1):
          pts = {p[0]: p for p in data[label]}
          common_xs = sorted(set(pts_ref.keys()) & set(pts.keys()))
          if ratio:
            vals = [pts[x][1] / pts_ref[x][1] if pts_ref[x][1] != 0 else float('nan') for x in common_xs]
          else:
            vals = [pts[x][1] - pts_ref[x][1] for x in common_xs]
          ax_bot.plot(common_xs, vals, marker='o', label=f'{label} / {ref_label}' if ratio else f'{label} − {ref_label}', color=colors[i % len(colors)])
        ax_bot.axhline(1.0 if ratio else 0.0, color='gray', linestyle='--', linewidth=0.8)
        ax_bot.set_ylabel(f'Ratio vs {ref_label}' if ratio else f'Diff vs {ref_label}')
        ax_bot.legend(fontsize='small')

      ax_bot.set_xticks(ticks)
      ax_bot.xaxis.set_major_locator(plt.FixedLocator(ticks))
      ax_bot.xaxis.set_major_formatter(plt.ScalarFormatter())
      ax_bot.set_xlabel('# Sampling Steps $S$')

      if col > 0:
        ax_top.set_ylabel('')
        ax_bot.set_ylabel('')

    handles, labels_leg = all_axes[0][0].get_legend_handles_labels()
    n_col_legend = len(labels) if n_col_legend is None else n_col_legend
    fig.legend(handles, labels_leg, loc='upper right', bbox_to_anchor=(0.91, 1), ncol=n_col_legend)

    return fig, all_axes


  def plot_quality_over_T_single(
      self,
      metric,
      agg='mean',
      xscale='log',
      baseline=None,
      ax=None,
      skip_colors=0,
      colors=None,
      show_uncertainty=True,
  ):
    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']

    if colors is None and skip_colors > 0:
      ax.set_prop_cycle(color=prop_cycle[skip_colors:])

    data = self._collect_quality_over_T_data(metric, agg=agg)
    labels = list(data.keys())

    # resolve colors per label
    label_colors = {}
    if colors is not None:
      for i, label in enumerate(labels):
        if isinstance(colors, dict):
          label_colors[label] = colors.get(label, prop_cycle[(i + skip_colors) % len(prop_cycle)])
        else:
          label_colors[label] = colors[i % len(colors)]

    for label, points in data.items():
      points.sort(key=lambda p: p[0])
      plot_kwargs = {}
      if label in label_colors:
        plot_kwargs['color'] = label_colors[label]
      ax.plot(
        [p[0] for p in points],
        [p[1] for p in points],
        marker='o',
        label=label,
        **plot_kwargs,
      )
      if show_uncertainty:
        ax.fill_between(
          [p[0] for p in points],
          [p[2] for p in points],
          [p[3] for p in points],
          alpha=0.2,
          **plot_kwargs,
        )

    if baseline is not None:
      y_baseline = baseline.volume_metrics[metric].mean() if agg == 'mean' else baseline.volume_metrics[metric].median()
      ax.axhline(y_baseline, color='black', linestyle='--', label='Baseline')

    dir_symbol = '↑' if is_higher_better(metric) else '↓'
    ax.set_title(get_metric_label(metric))
    ax.text(0.1, -0.155, f'{dir_symbol} is better', transform=ax.transAxes,
            ha='right', va='bottom', fontsize='medium', color='gray')
    ax.set_xlabel('Sampling $S$')
    ax.set_xscale(xscale)
    ticks = sorted({p[0] for points in data.values() for p in points})
    ax.set_xticks(ticks)
    ax.xaxis.set_major_locator(plt.FixedLocator(ticks))
    ax.xaxis.set_major_formatter(plt.ScalarFormatter())

    if show_uncertainty:
      ax.set_ylabel(f'{agg.capitalize()}/{ "std" if agg == "mean" else "IQR" } ' + get_metric_abbr(metric))
    else:
      ax.set_ylabel(f'{agg.capitalize()} ' + get_metric_abbr(metric))

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

  def visualise_slice_comparison_with_ct(
      self,
      sample,
      slice_index,
      axis='sagittal',
      dose_cmap='jet',
      dose_alpha=0.5,
      show_oar=True,
      show_beams=False,
      crop_to_body=False,
      crop_padding=2,
      max_error=None,
      dose_range=None,
      n_error_levels=11,
      fig=None,
      axes=None,
      add_colorbar=True,
      show_row_labels=True,
  ):
    """
    Visualize slice comparison with CT underneath transparent dose distributions.

    Args:
        sample: Sample name to visualize
        slice_index: Index of the slice to visualize
        axis: 'sagittal', 'axial', or 'coronal'
        dose_cmap: Colormap for dose overlay (default: 'jet')
        dose_alpha: Transparency of dose overlay (default: 0.5)
        show_oar: Whether to show OAR contours (default: True)
        crop_to_body: Whether to crop the view to the body region (default: False)
        crop_padding: Padding in pixels around the body bounding box (default: 5)
        max_error: Maximum error for error colorbar (default: None)
        dose_range: Range for dose colorbar (default: None)
        n_error_levels: Number of discrete levels for error colorbar (default: 11)

    Returns:
        Tuple of (figure, axes)
    """
    col_width = 3 if crop_to_body else 4
    if axes is not None:
      pass  # use caller-provided axes; fig must also be provided for colorbars
    elif fig is None:
      fig, axes = plt.subplots(2, len(self.results_list) + 1, figsize=(col_width * (len(self.results_list) + 1), 6.4), constrained_layout=True)
    else:
      axes = fig.subplots(2, len(self.results_list) + 1)

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

    # Get beam plate if needed
    beam_plate = None
    if show_beams:
      beam_plate = first_result.dataset.selector.get_component(structure, 'beam_plate')

    if dose_range is not None:
      dose_min, dose_max = dose_range
    else:
      dose_min, dose_max = 0, original_dose.max()

    # Pre-compute global error_max across all models for consistent error colorbar
    if max_error is not None:
      global_error_max = max_error
    else:
      global_error_max = 0
      for result in self.results_list:
        sr = result.for_sample(sample)
        pred_vol = sr.load_pred_volume()
        if axis == "sagittal":
          pred = np.flip(pred_vol[:, :, slice_index])
        elif axis == "axial":
          pred = pred_vol[slice_index, :, :]
        else:
          pred = np.flip(pred_vol[:, slice_index, :], 0)
        global_error_max = max(global_error_max, np.max(np.abs(original_dose - pred)))

    # Row 0, Col 0: CT with original dose overlay
    axes[0, 0].imshow(ct_slice, cmap='gray')
    dose_masked = np.ma.masked_where(original_dose < 0.01 * dose_max, original_dose)
    im_dose = axes[0, 0].imshow(dose_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=dose_min, vmax=dose_max)
    draw_ptv_contours(axes[0, 0], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(axes[0, 0], oar_masks)
    axes[0, 0].set_title('Ground Truth')
    if show_row_labels:
      axes[0, 0].text(-0.05, 0.5, 'Dose', transform=axes[0, 0].transAxes, fontsize=12, rotation=90, va='center', ha='right')
    axes[0, 0].axis('off')

    # Row 1, Col 0: CT with PTV only (reference)
    axes[1, 0].imshow(ct_slice, cmap='gray')
    if show_beams and beam_plate is not None:
      draw_beam_overlay(axes[1, 0], beam_plate)
    draw_ptv_overlay(axes[1, 0], ptv_doses)
    draw_ptv_contours(axes[1, 0], ptv_doses)
    if show_oar and oar_masks is not None:
      draw_oar_contours(axes[1, 0], oar_masks)
    axes[1, 0].set_title('')
    if show_row_labels:
      axes[1, 0].text(-0.05, 0.5, 'Structures', transform=axes[1, 0].transAxes, fontsize=12, rotation=90, va='center', ha='right')
    axes[1, 0].axis('off')

    # Discretize error colorbar
    error_boundaries = np.linspace(-global_error_max, global_error_max, n_error_levels + 1)
    error_norm = BoundaryNorm(error_boundaries, plt.get_cmap('bwr').N)

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
      axes[0, i + 1].imshow(pred_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=dose_min, vmax=dose_max)
      draw_ptv_contours(axes[0, i + 1], ptv_doses)
      if show_oar and oar_masks is not None:
        draw_oar_contours(axes[0, i + 1], oar_masks)
      axes[0, i + 1].set_title(self.result_labels[i])
      axes[0, i + 1].axis('off')

      # Row 1: CT with error overlay
      error = original_dose - pred_dose
      axes[1, i + 1].imshow(ct_slice, cmap='gray')
      error_masked = np.ma.masked_where(np.abs(error) < 0.01 * global_error_max, error)
      im_err = axes[1, i + 1].imshow(error_masked, cmap='bwr', alpha=dose_alpha, norm=error_norm)
      draw_ptv_contours(axes[1, i + 1], ptv_doses)
      if show_oar and oar_masks is not None:
        draw_oar_contours(axes[1, i + 1], oar_masks)
      axes[1, i + 1].axis('off')

    # Row labels on the left side of the generated columns
    if show_row_labels:
      axes[0, 1].text(-0.03, 0.5, 'Generated', transform=axes[0, 1].transAxes, fontsize=12, rotation=90, va='center', ha='right')
      axes[1, 1].text(-0.03, 0.5, 'Error', transform=axes[1, 1].transAxes, fontsize=12, rotation=90, va='center', ha='right')

    # Shared colorbars: one for dose (row 0), one for error (row 1)
    if add_colorbar:
      fig.colorbar(im_dose, ax=axes[0, :].tolist(), fraction=0.02, pad=0.005, shrink=0.8, label='Dose [Gy]')
      fig.colorbar(im_err, ax=axes[1, :].tolist(), fraction=0.02, pad=0.005, shrink=0.8, label='Error [Gy]')

    # Crop to body bounding box if requested
    if crop_to_body:
      body_mask = first_result.dataset.selector.get_body(structure)[0] > 0.5
      rows = np.any(body_mask, axis=1)
      cols = np.any(body_mask, axis=0)
      rmin, rmax = np.where(rows)[0][[0, -1]]
      cmin, cmax = np.where(cols)[0][[0, -1]]
      # Apply padding
      rmin = max(0, rmin - crop_padding)
      rmax = min(ct_slice.shape[0] - 1, rmax + crop_padding)
      cmin = max(0, cmin - crop_padding)
      cmax = min(ct_slice.shape[1] - 1, cmax + crop_padding)
      for ax_row in axes:
        for ax in ax_row:
          ax.set_xlim(cmin, cmax)
          ax.set_ylim(rmax, rmin)

    return fig, axes, im_dose, im_err

  def visualise_uncertainty_with_ct(
      self,
      slices,
      axis='sagittal',
      dose_cmap='jet',
      dose_alpha=0.5,
      show_oar=True,
      show_beams=False,
      crop_to_body=False,
      crop_padding=2,
      dose_range=None,
      max_error=None,
      n_error_levels=11,
      error_cmap='hot',
  ):
    """
    Visualize uncertainty with CT underneath.
    Shows per row: structures, ground truth, best prediction, worst prediction, stddev, RMSE.

    Args:
        slices: List of (sample, slice_index) tuples, one row per entry.
        axis: 'sagittal', 'axial', or 'coronal'
        dose_cmap: Colormap for dose overlay (default: 'jet')
        dose_alpha: Transparency of dose overlay (default: 0.5)
        show_oar: Whether to show OAR contours (default: True)
        show_beams: Whether to show beam plate overlay on structure column (default: False)
        crop_to_body: Whether to crop the view to the body region (default: False)
        crop_padding: Padding in pixels around the body bounding box (default: 2)
        dose_range: Range for dose colorbar (default: None)
        max_error: Maximum error for stddev/MAE colorbar (default: None, auto)
        n_error_levels: Number of discrete levels for stddev/MAE colorbar (default: 11)
        error_cmap: Colormap for stddev and MAE overlays (default: 'hot')

    Returns:
        Tuple of (figure, axes)
    """
    n_preds = len(self.results_list)
    n_rows = len(slices)
    row_height = 2.4
    cbar_height_ratios = [1] * n_rows + [0.04]

    fig, all_axes = plt.subplots(
      n_rows + 1, 6,
      figsize=(12, row_height * n_rows + 0.6),
      gridspec_kw={'height_ratios': cbar_height_ratios},
      constrained_layout=True,
      squeeze=False,
    )
    axes = all_axes[:n_rows]
    cbar_axes = all_axes[n_rows]
    # Hide the structure column's colorbar cell
    cbar_axes[0].axis('off')

    first_result = self.results_list[0]

    # Pre-compute all row data to determine global ranges
    row_data = []
    for sample, slice_index in slices:
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

      ct_slice = first_result.dataset.selector.get_ct(structure)[0]
      ptv_doses = first_result.dataset.selector.get_component(structure, 'ptv_doses')
      oar_masks = None
      if show_oar:
        oar_masks = first_result.dataset.selector.get_component(structure, 'oar_v3')
      beam_plate = None
      if show_beams:
        beam_plate = first_result.dataset.selector.get_component(structure, 'beam_plate')
      body = first_result.dataset.selector.get_body(structure)[0] if crop_to_body else None

      preds = []
      for result in self.results_list:
        sr = result.for_sample(sample)
        pred_vol = sr.load_pred_volume()
        if axis == "sagittal":
          pred = np.flip(pred_vol[:, :, slice_index])
        elif axis == "axial":
          pred = pred_vol[slice_index, :, :]
        else:
          pred = np.flip(pred_vol[:, slice_index, :], 0)
        preds.append(pred)

      std_pred = np.std(preds, axis=0)
      per_pred_rmse = [np.sqrt(np.mean((original_dose - pred) ** 2)) for pred in preds]
      best_idx = int(np.argmin(per_pred_rmse))
      worst_idx = int(np.argmax(per_pred_rmse))
      rmse = np.sqrt(np.mean([(original_dose - pred) ** 2 for pred in preds], axis=0))

      row_data.append({
        'sample': sample,
        'slice_index': slice_index,
        'ct_slice': ct_slice,
        'ptv_doses': ptv_doses,
        'oar_masks': oar_masks,
        'beam_plate': beam_plate,
        'body': body,
        'original_dose': original_dose,
        'best_pred': preds[best_idx],
        'best_label': self.result_labels[best_idx],
        'worst_pred': preds[worst_idx],
        'worst_label': self.result_labels[worst_idx],
        'std_pred': std_pred,
        'rmse': rmse,
      })

    # Compute global dose range
    if dose_range is not None:
      dose_min, dose_max = dose_range
    else:
      dose_min = 0
      dose_max = max(d['original_dose'].max() for d in row_data)

    # Compute global error max for stddev/MAE colorbar
    if max_error is not None:
      global_error_max = max_error
    else:
      global_error_max = max(
        max(np.max(d['std_pred']) for d in row_data),
        max(np.max(d['rmse']) for d in row_data),
      )

    error_boundaries = np.linspace(0, global_error_max, n_error_levels + 1)
    error_norm = BoundaryNorm(error_boundaries, plt.get_cmap(error_cmap).N)

    im_dose = None
    im_err = None

    for row_idx, d in enumerate(row_data):
      ax_row = axes[row_idx]
      ct_slice = d['ct_slice']
      ptv_doses = d['ptv_doses']
      oar_masks = d['oar_masks']

      def _draw_contours(ax, _ptv=ptv_doses, _oar=oar_masks):
        draw_ptv_contours(ax, _ptv)
        if show_oar and _oar is not None:
          draw_oar_contours(ax, _oar)

      # Col 0: Structures
      ax_row[0].imshow(ct_slice, cmap='gray')
      if show_beams and d['beam_plate'] is not None:
        draw_beam_overlay(ax_row[0], d['beam_plate'])
      draw_ptv_overlay(ax_row[0], ptv_doses)
      _draw_contours(ax_row[0])
      ax_row[0].axis('off')

      # Col 1: Ground truth
      ax_row[1].imshow(ct_slice, cmap='gray')
      dose_masked = np.ma.masked_where(d['original_dose'] < 0.01 * dose_max, d['original_dose'])
      im_dose = ax_row[1].imshow(dose_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=dose_min, vmax=dose_max)
      _draw_contours(ax_row[1])
      ax_row[1].axis('off')

      # Col 2: Best prediction
      ax_row[2].imshow(ct_slice, cmap='gray')
      best_masked = np.ma.masked_where(d['best_pred'] < 0.01 * dose_max, d['best_pred'])
      ax_row[2].imshow(best_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=dose_min, vmax=dose_max)
      _draw_contours(ax_row[2])
      ax_row[2].axis('off')

      # Col 3: Worst prediction
      ax_row[3].imshow(ct_slice, cmap='gray')
      worst_masked = np.ma.masked_where(d['worst_pred'] < 0.01 * dose_max, d['worst_pred'])
      ax_row[3].imshow(worst_masked, cmap=dose_cmap, alpha=dose_alpha, vmin=dose_min, vmax=dose_max)
      _draw_contours(ax_row[3])
      ax_row[3].axis('off')

      # Col 4: Stddev
      ax_row[4].imshow(ct_slice, cmap='gray')
      std_masked = np.ma.masked_where(d['std_pred'] < 0.01 * global_error_max, d['std_pred'])
      im_err = ax_row[4].imshow(std_masked, cmap=error_cmap, alpha=dose_alpha, norm=error_norm)
      _draw_contours(ax_row[4])
      ax_row[4].axis('off')

      # Col 5: RMSE
      ax_row[5].imshow(ct_slice, cmap='gray')
      rmse_masked = np.ma.masked_where(d['rmse'] < 0.01 * global_error_max, d['rmse'])
      ax_row[5].imshow(rmse_masked, cmap=error_cmap, alpha=dose_alpha, norm=error_norm)
      _draw_contours(ax_row[5])
      ax_row[5].axis('off')

      # Row label
      ax_row[0].set_ylabel(f"{d['sample']}:{d['slice_index']}", fontsize=10)

      # Crop to body bounding box if requested
      if crop_to_body:
        body_mask = d['body'] > 0.5
        rows = np.any(body_mask, axis=1)
        cols = np.any(body_mask, axis=0)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmin = max(0, rmin - crop_padding)
        rmax = min(ct_slice.shape[0] - 1, rmax + crop_padding)
        cmin = max(0, cmin - crop_padding)
        cmax = min(ct_slice.shape[1] - 1, cmax + crop_padding)
        for ax in ax_row:
          ax.set_xlim(cmin, cmax)
          ax.set_ylim(rmax, rmin)

    # Column titles (top row only)
    axes[0, 0].set_title('Structures')
    axes[0, 1].set_title('Ground Truth')
    axes[0, 2].set_title('Best Generation')
    axes[0, 3].set_title('Worst Generation')
    axes[0, 4].set_title(f'Std Dev (n={n_preds})')
    axes[0, 5].set_title(f'RMSE (n={n_preds})')

    fig.suptitle('Uncertainty Visualization with CT')

    # Colorbars (horizontal, in dedicated bottom row)
    # Merge cells 1–3 for dose colorbar
    gs = all_axes[0, 0].get_gridspec()
    for ax in cbar_axes[1:4]:
      ax.remove()
    cax_dose = fig.add_subplot(gs[n_rows, 1:4])
    fig.colorbar(im_dose, cax=cax_dose, orientation='horizontal', label='Dose [Gy]')

    # Merge cells 4–5 for error colorbar
    for ax in cbar_axes[4:6]:
      ax.remove()
    cax_err = fig.add_subplot(gs[n_rows, 4:6])
    fig.colorbar(im_err, cax=cax_err, orientation='horizontal', label='Dose Difference [Gy]')

    return fig, axes

  def plot_rank_frequencies(self, metrics, ax=None):
    n_models = len(self.results_list)
    ranks = np.arange(1, n_models + 1)

    # Collect metric values per model per sample
    # Shape: (n_metrics, n_samples, n_models)
    all_ranks = []
    for metric in metrics:
      higher_better = is_higher_better(metric)
      values = []
      for result in self.results_list:
        values.append(result.volume_metrics[metric].values)
      # values: (n_models, n_samples)
      values = np.array(values)
      n_samples = values.shape[1]

      # Rank per sample (axis=0 ranks across models)
      # rankdata gives rank 1=smallest, so for higher_better we negate
      sample_ranks = np.zeros_like(values, dtype=int)
      for s in range(n_samples):
        col = values[:, s]
        if higher_better:
          col = -col
        sample_ranks[:, s] = rankdata(col, method='min').astype(int)
      all_ranks.append(sample_ranks)

    # Average ranks across all metrics and samples per model
    # all_ranks: list of (n_models, n_samples) arrays
    combined_ranks = np.concatenate(all_ranks, axis=1)  # (n_models, n_metrics * n_samples)

    # Build frequency matrix: (n_models, n_ranks)
    freq_matrix = np.zeros((n_models, n_models), dtype=int)
    for m in range(n_models):
      for rank in ranks:
        freq_matrix[m, rank - 1] = np.sum(combined_ranks[m] == rank)

    # Sort models by 1st rank frequency (desc), then 2nd rank, etc.
    sort_keys = [-freq_matrix[:, r] for r in range(n_models)]
    sort_order = np.lexsort(sort_keys[::-1])
    freq_matrix = freq_matrix[sort_order]
    sorted_labels = [self.result_labels[i] for i in sort_order]
    sorted_groups = [self.result_groups[i] for i in sort_order]

    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(max(6, n_models * 0.8), max(4, n_models * 0.5)))

    # Create color-coded heatmap using groups
    if self.has_groups:
      unique_groups = list(dict.fromkeys(self.result_groups))  # preserve order
      prop_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
      group_to_color = {g: prop_cycle[i % len(prop_cycle)] for i, g in enumerate(unique_groups)}

      # Create an RGBA image where each cell's hue comes from its group
      # and alpha/intensity from the frequency
      max_freq = freq_matrix.max()
      rgba_image = np.ones((n_models, n_models, 4))
      for m in range(n_models):
        base_color = to_rgba(group_to_color[sorted_groups[m]])
        for r in range(n_models):
          intensity = freq_matrix[m, r] / max_freq if max_freq > 0 else 0
          # Blend white -> group color based on intensity
          rgba_image[m, r, :3] = 1 - intensity * (1 - np.array(base_color[:3]))
          rgba_image[m, r, 3] = 1.0

      ax.imshow(rgba_image, aspect='auto')

      # Add text annotations
      for m in range(n_models):
        for r in range(n_models):
          val = freq_matrix[m, r]
          text_color = 'white' if val / max_freq > 0.6 else 'black'
          ax.text(r, m, str(val), ha='center', va='center', fontsize=9, color=text_color)

      # Add legend for groups
      legend_handles = [Patch(facecolor=group_to_color[g], label=g) for g in unique_groups]
      ax.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(1.02, 1.09), ncol=2, title='')
    else:
      sns.heatmap(freq_matrix, ax=ax, annot=True, fmt='d', cmap='Blues',
                  xticklabels=ranks, yticklabels=sorted_labels, cbar_kws={'label': 'Frequency'})

    ax.set_xticks(np.arange(n_models) + (0.5 if not self.has_groups else 0))
    ax.set_xticklabels(ranks)
    ax.set_yticks(np.arange(n_models) + (0.5 if not self.has_groups else 0))
    ax.set_yticklabels(sorted_labels)
    ax.set_xlabel('Ranks')
    ax.set_ylabel('')
    ax.grid(False)
    ax.set_title(f'Rank Frequencies')

    return ax

  def plot_metric_heatmap(self, metrics, baseline: 'ModelComparison', improved_q=0.75, worsened_q=0.25, sort=True, baseline_label='Baseline', ax=None):
    """
    Heatmap with median metric values per model. Annotations show how many subjects
    improved or worsened compared to subject-specific baseline percentiles.

    Args:
        metrics: List of metric column names.
        baseline: A ModelComparison whose runs define the per-subject reference distribution.
        improved_q: Percentile threshold for improvement (default 0.75).
        worsened_q: Percentile threshold for worsening (default 0.25).
        sort: Whether to sort models by average rank across metrics.
        ax: Optional matplotlib axes.
    """
    # Build per-subject baseline thresholds from the baseline ModelComparison
    baseline_volumes = pd.concat(
      [r.volume_metrics for r in baseline.results_list], ignore_index=True
    )

    baseline_thresholds = {}
    for metric in metrics:
      grouped = baseline_volumes.groupby('file_name')[metric]
      baseline_thresholds[metric] = pd.DataFrame({
        'q_improved': grouped.quantile(improved_q),
        'q_worsened': grouped.quantile(worsened_q),
      })

    # Compute median, improved count, and worsened count per model per metric
    model_labels = []
    median_matrix = []
    annot_matrix = []

    for result, label in zip(self.results_list, self.result_labels):
      model_labels.append(label)
      medians = []
      annots = []
      for metric in metrics:
        vm = result.volume_metrics.set_index('file_name')
        thresholds = baseline_thresholds[metric]

        # Only compare subjects present in both
        common = vm.index.intersection(thresholds.index)
        values = vm.loc[common, metric]
        q_imp = thresholds.loc[common, 'q_improved']
        q_wor = thresholds.loc[common, 'q_worsened']

        higher_better = is_higher_better(metric)
        if higher_better:
          n_improved = (values > q_imp).sum()
          n_worsened = (values < q_wor).sum()
        else:
          n_improved = (values < q_wor).sum()
          n_worsened = (values > q_imp).sum()

        medians.append(result.volume_metrics[metric].median())
        annots.append(f'\u2191{n_improved} \u2193{n_worsened}')

      median_matrix.append(medians)
      annot_matrix.append(annots)

    col_labels = [get_metric_abbr(m, True, with_unit=False) for m in metrics]
    metric_units = [get_metric_unit(m) for m in metrics]
    median_df = pd.DataFrame(median_matrix, index=model_labels, columns=col_labels)
    annot_df = pd.DataFrame(annot_matrix, index=model_labels, columns=col_labels)

    # Compute baseline median row
    baseline_medians = [
      pd.Series([r.volume_metrics[m].median() for r in baseline.results_list]).median()
      for m in metrics
    ]

    # Sort models by average normalised rank across metrics
    if sort:
      ranks = pd.DataFrame(index=median_df.index)
      for i, metric in enumerate(metrics):
        col = median_df.columns[i]
        ascending = not is_higher_better(metric)
        ranks[col] = median_df[col].rank(ascending=ascending, method='min')
      order = ranks.mean(axis=1).sort_values().index
      median_df = median_df.loc[order]
      annot_df = annot_df.loc[order]

    # Prepend baseline row, then normalise all medians together for colouring
    gap_height = 0.15
    baseline_median_row = pd.DataFrame([baseline_medians], index=[baseline_label], columns=col_labels)
    combined_median = pd.concat([baseline_median_row, median_df])

    # Normalise so that baseline = 0, better = positive, worse = negative,
    # then map to [0, 1] with 0.5 = baseline for a diverging colormap.
    combined_norm = combined_median.copy()
    for i, metric in enumerate(metrics):
      col = combined_median.columns[i]
      baseline_val = baseline_medians[i]
      diff = combined_median[col] - baseline_val
      if not is_higher_better(metric):
        diff = -diff  # flip so positive = better
      max_abs = diff.abs().max()
      if max_abs > 0:
        scaled = np.sign(diff) * np.sqrt(diff.abs() / max_abs)
        combined_norm[col] = 0.5 + 0.5 * scaled
      else:
        combined_norm[col] = 0.5

    if ax is None:
      fig, ax = plt.subplots(1, 1, figsize=(len(metrics) * 1 + 2, (len(model_labels) + 1) * 0.7 + 0.5))

    # Custom y positions and heights: baseline shorter, then gap, then model rows
    baseline_height = 0.6
    row_heights = [baseline_height] + [1] * len(model_labels)
    y_positions = [0]  # baseline
    for k in range(len(model_labels)):
      y_positions.append(baseline_height + gap_height + k)
    total_height = y_positions[-1] + 1

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('ggplot_diverge', ['#E24A33', 'white', '#348ABD'])
    ax.set_facecolor('none')
    ax.get_figure().patch.set_facecolor('white')

    # Draw heatmap cells manually
    for row_idx in range(len(combined_norm)):
      y = y_positions[row_idx]
      h = row_heights[row_idx]
      for j in range(len(metrics)):
        norm_val = combined_norm.iloc[row_idx, j]
        if pd.isna(norm_val):
          continue
        color = cmap(norm_val)
        ax.add_patch(plt.Rectangle((j, y), 1, h, facecolor=color, edgecolor='white', linewidth=0.5))

    ax.set_xlim(0, len(metrics))
    ax.set_ylim(total_height, 0)
    ax.set_xticks([j + 0.5 for j in range(len(metrics))])
    ax.set_xticklabels(col_labels, rotation=0)
    ax.set_yticks([y + h / 2 for y, h in zip(y_positions, row_heights)])
    ylabels = list(combined_norm.index)
    ax.set_yticklabels(ylabels, rotation=0)
    ax.get_yticklabels()[0].set_fontweight('bold')

    # Draw annotations with adaptive text color
    def _median_text(val, unit):
      unit_str = f' {unit}' if unit else ''
      if val != 0 and abs(val) < 0.1 :
        return f'$\\tilde{{x}}$ = {val:.3f}{unit_str}'
      return f'$\\tilde{{x}}$ = {val:.2f}{unit_str}'

    for row_idx in range(len(combined_norm)):
      y = y_positions[row_idx]
      h = row_heights[row_idx]
      for j in range(len(metrics)):
        val = combined_median.iloc[row_idx, j]
        norm_val = combined_norm.iloc[row_idx, j]
        if pd.isna(val):
          continue
        rgba = cmap(norm_val)
        luminance = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
        text_color = 'white' if luminance < 0.5 else 'black'
        unit = metric_units[j]
        if row_idx == 0:  # baseline row: only median, centered
          ax.text(j + 0.5, y + h / 2, _median_text(val, unit),
                  ha='center', va='center', fontsize=7, color=text_color, fontweight='bold')
        else:  # model rows: median + arrow counts
          ax.text(j + 0.5, y + h * 0.33, _median_text(val, unit),
                  ha='center', va='center', fontsize=7, color=text_color)
          ax.text(j + 0.5, y + h * 0.67, annot_df.iloc[row_idx - 1, j],
                  ha='center', va='center', fontsize=11, color=text_color)

    # Remove default spines
    for spine in ax.spines.values():
      spine.set_visible(False)
    ax.tick_params(left=False, bottom=False)

    ax.set_title(f'Model Improvement or Worsening Compared to Baseline \n(↑ improved count, ↓ worsened count)')
    ax.set_ylabel('')

    return ax

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
