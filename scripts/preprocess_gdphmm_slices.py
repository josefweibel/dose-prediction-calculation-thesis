#!/usr/bin/env python3
"""
Preprocessing script for GDPHMMSlicesDataset.

This script processes 3D volumes and saves them as individual 2D slices,
ready to be loaded directly without on-the-fly preprocessing.

All transformations from the original dataset are applied:
  - CT: clipped to [-1024, 3071], normalized to [0, 1], stored as float32
  - Dose: normalized to [0, 1], stored as float32
  - PTV: dose levels {0, 56, 63, 70}, stored as uint8 (saves 75% vs float32)
  - Body: stored as uint8
  - OAR: binary masks from SDM, stored as uint8
  - OAR SDM: signed distance maps, stored as float16 (saves 50% space)
  - PTV SDM: signed distance map, stored as float16 (saves 50% space)

IMPORTANT: SDMs are computed AFTER padding to ensure correct distance values at edges.
If SDMs were computed before padding, the padded regions would have incorrect 0 values.

Output structure:
  data/{dataset_name}_slices/{split}/{volume_name}/slice_{idx:04d}.npz

Each slice contains:
  - ct: (1, H, W) float32 - clipped and normalized CT scan
  - dose: (1, H, W) float32 - normalized dose distribution
  - ptv: (1, H, W) uint8 - PTV dose levels {0, 56, 63, 70} Gy
  - body: (1, H, W) uint8 - body mask
  - oar: (N_OAR, H, W) uint8 - binary OAR masks
  - oar_sdm: (N_OAR, H, W) float16 - signed distance maps for OARs
  - ptv_sdm: (1, H, W) float16 - signed distance map for PTV
"""

import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
from scipy.ndimage import distance_transform_edt
from multiprocessing import Pool, cpu_count
import sys

# Import constants from data module
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from data import (
  GDPHMM_DATA_BASE_PATH,
  GDPHMM_IMG_MIN,
  GDPHMM_IMG_MAX,
  GDPHMM_DOSE_MIN,
  GDPHMM_DOSE_MAX,
  GDPHMM_OAR,
  GDPHMM_PTV,
  compute_sdm
)


def compute_sdms(oar_vol, ptv_vol, voxel_spacing):
  """
  Compute signed distance maps for OAR and PTV masks.

  Args:
    oar_vol: (N_OAR, H, W, D) - binary masks for organs at risk
    ptv_vol: (H, W, D) - PTV mask with dose values
    voxel_spacing: tuple of (sx, sy, sz) - voxel spacing in mm

  Returns:
    oar_sdm: (N_OAR, H, W, D) float32 - SDMs for each OAR
    ptv_sdm: (H, W, D) float32 - SDM for PTV
  """
  oar_sdm = np.stack([
    compute_sdm(oar_vol[i], voxel_spacing)
    for i in range(oar_vol.shape[0])
  ], axis=0)

  ptv_binary = (ptv_vol > 0).astype(np.uint8)
  ptv_sdm = compute_sdm(ptv_binary, voxel_spacing)

  return oar_sdm, ptv_sdm


def apply_transformations(ct_vol, dose_vol, ptv_vol, body_vol, oar_vol, pad_size=None):
  """
  Apply all transformations to match the GDPHMMSlicesDataset behavior.

  This includes:
    - CT: clip to [-1024, 3071], normalize to [0, 1]
    - Dose: normalize to [0, 1]
    - PTV: store dose levels as uint8 (0, 56, 63, 70) instead of float32 (0.0, 0.56, 0.63, 0.70)
    - Body: convert to uint8
    - OAR: convert to uint8
    - Padding: optionally pad to fixed size (saves runtime padding)

  Args:
    ct_vol: (H, W, D) float - CT scan
    dose_vol: (H, W, D) float - dose distribution
    ptv_vol: (H, W, D) float - PTV mask with values in [0, 1] (dose/100)
    body_vol: (H, W, D) - body mask
    oar_vol: (N_OAR, H, W, D) - OAR binary masks
    pad_size: optional (H, W) to pad all slices to this size

  Returns:
    Transformed versions ready for direct use (without SDMs - those are computed after padding)
  """
  # CT: clip and normalize
  ct_transformed = np.clip(ct_vol, GDPHMM_IMG_MIN, GDPHMM_IMG_MAX)
  ct_transformed = (ct_transformed - GDPHMM_IMG_MIN) / (GDPHMM_IMG_MAX - GDPHMM_IMG_MIN)
  ct_transformed = ct_transformed.astype(np.float32)

  # Dose: normalize
  dose_transformed = (dose_vol - GDPHMM_DOSE_MIN) / (GDPHMM_DOSE_MAX - GDPHMM_DOSE_MIN)
  dose_transformed = dose_transformed.astype(np.float32)

  # PTV: store as dose levels (0, 56, 63, 70) instead of normalized (0.0, 0.56, 0.63, 0.70)
  # This saves 75% storage (float32 -> uint8) with no information loss
  ptv_transformed = np.round(ptv_vol * 100).astype(np.uint8)

  # Body: convert to uint8
  body_transformed = body_vol.astype(np.uint8)

  # OAR: convert to uint8
  oar_transformed = oar_vol.astype(np.uint8)

  # Optionally pad all volumes to fixed size using np.pad (much simpler and faster!)
  if pad_size is not None:
    target_h, target_w = pad_size
    current_h, current_w = ct_transformed.shape[:2]

    # Calculate padding amounts (symmetric)
    pad_h = target_h - current_h
    pad_w = target_w - current_w

    if pad_h < 0 or pad_w < 0:
      raise ValueError(f"Volume size ({current_h}, {current_w}) is larger than target size {pad_size}")

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    # Pad volumes directly using np.pad
    # Format: ((before_axis0, after_axis0), (before_axis1, after_axis1), (before_axis2, after_axis2))
    pad_2d = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))  # Don't pad depth dimension
    pad_3d = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))  # For OAR arrays (N_OAR, H, W, D)

    ct_transformed = np.pad(ct_transformed, pad_2d, mode='constant', constant_values=0)
    dose_transformed = np.pad(dose_transformed, pad_2d, mode='constant', constant_values=0)
    ptv_transformed = np.pad(ptv_transformed, pad_2d, mode='constant', constant_values=0)
    body_transformed = np.pad(body_transformed, pad_2d, mode='constant', constant_values=0)
    oar_transformed = np.pad(oar_transformed, pad_3d, mode='constant', constant_values=0)

  return ct_transformed, dose_transformed, ptv_transformed, body_transformed, oar_transformed


def process_volume(args):
  """
  Process a single volume file and save all its slices.

  Args:
    args: tuple of (file_path, output_dir, voxel_spacing, pad_size)

  Returns:
    tuple of (volume_name, num_slices, success)
  """
  file_path, output_dir, voxel_spacing, pad_size = args

  try:
    # Load volume
    vol = np.load(file_path, allow_pickle=True)['arr_0'].item()

    # Extract base data
    dose_vol = (vol['dose'] * vol['dose_scale'] * vol['Body'])
    ct_vol = vol['img']
    body_vol = vol['Body']

    # Build OAR volume (N_OAR, H, W, D)
    oar_vol = np.stack(
      [vol[name].astype(np.uint8) if name in vol else np.zeros_like(vol['dose'].astype(np.uint8))
       for name in GDPHMM_OAR],
      axis=0
    )

    # Build PTV volume (H, W, D)
    ptv_vol = np.maximum.reduce(
      [vol[name] * dose / 100 for dose, name in GDPHMM_PTV.items() if name in vol]
    )

    # Apply all transformations (clipping, normalization, type conversion, padding)
    # NOTE: Padding is applied BEFORE SDM computation to avoid 0-valued edges in distance maps
    ct_vol, dose_vol, ptv_vol, body_vol, oar_vol = apply_transformations(
      ct_vol, dose_vol, ptv_vol, body_vol, oar_vol, pad_size=pad_size
    )

    # Compute signed distance maps AFTER padding so distances are correct at edges
    oar_sdm, ptv_sdm = compute_sdms(oar_vol, ptv_vol, voxel_spacing)

    # Convert SDMs to float16 for storage efficiency (saves 50% space, sufficient precision)
    oar_sdm = oar_sdm.astype(np.float16)
    ptv_sdm = ptv_sdm.astype(np.float16)

    # Create output directory for this volume
    volume_name = file_path.stem
    volume_dir = output_dir / volume_name
    volume_dir.mkdir(parents=True, exist_ok=True)

    # Save each slice with channel dimension
    num_slices = ct_vol.shape[2]
    for slice_idx in range(num_slices):
      slice_data = {
        'ct': ct_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'dose': dose_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'ptv': ptv_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'body': body_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'oar': oar_vol[:, :, :, slice_idx],  # (N_OAR, H, W)
        'oar_sdm': oar_sdm[:, :, :, slice_idx],  # (N_OAR, H, W)
        'ptv_sdm': ptv_sdm[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
      }

      slice_path = volume_dir / f"slice_{slice_idx:04d}.npz"
      np.savez_compressed(slice_path, **slice_data)

    return (volume_name, num_slices, True)

  except Exception as e:
    print(f"Error processing {file_path.name}: {e}")
    import traceback
    traceback.print_exc()
    return (file_path.name, 0, False)


def create_slice_index(output_dir, split):
  """
  Create an index CSV file mapping each slice to its file location.

  Args:
    output_dir: Path to preprocessed data directory
    split: train/valid/test

  Returns:
    DataFrame with columns: volume, slice_idx, file
  """
  split_dir = output_dir / split

  records = []
  for volume_dir in sorted(split_dir.iterdir()):
    if not volume_dir.is_dir():
      continue

    volume_name = volume_dir.name
    for slice_file in sorted(volume_dir.glob("slice_*.npz")):
      # Extract slice index from filename
      slice_idx = int(slice_file.stem.split('_')[1])

      # Store relative path
      rel_path = slice_file.relative_to(split_dir)

      records.append({
        'volume': volume_name,
        'slice_idx': slice_idx,
        'file': str(rel_path)
      })

  df = pd.DataFrame(records)
  index_path = split_dir / '_index.csv'
  df.to_csv(index_path, index=False)

  print(f"Created index with {len(df)} slices: {index_path}")
  return df


def preprocess_split(dataset_name, split, voxel_spacing, num_workers, pad_size=None, limit=None):
  """
  Preprocess all volumes in a dataset split.

  Args:
    dataset_name: Name of dataset (e.g., 'gdphmm-hn')
    split: train/valid/test
    voxel_spacing: tuple of (sx, sy, sz)
    num_workers: number of parallel workers
    pad_size: optional (H, W) to pad slices to fixed size
    limit: optionally limit number of volumes to process
  """
  print(f"\nProcessing {dataset_name}/{split}...")

  # Setup paths
  input_dir = GDPHMM_DATA_BASE_PATH / dataset_name / split
  output_base = GDPHMM_DATA_BASE_PATH / f"{dataset_name}_slices"
  output_dir = output_base / split
  output_dir.mkdir(parents=True, exist_ok=True)

  # Get list of volume files
  volume_files = sorted(input_dir.glob("*.npz"))
  volume_files = [f for f in volume_files if f.name != '_index.csv']

  if limit:
    volume_files = volume_files[:limit]

  print(f"Found {len(volume_files)} volumes to process")

  # Prepare arguments for parallel processing
  args_list = [(f, output_dir, voxel_spacing, pad_size) for f in volume_files]

  # Process volumes in parallel
  if num_workers > 1:
    with Pool(num_workers) as pool:
      results = list(tqdm(
        pool.imap(process_volume, args_list),
        total=len(args_list),
        desc=f"Processing {split}"
      ))
  else:
    results = [process_volume(args) for args in tqdm(args_list, desc=f"Processing {split}")]

  # Summary
  successful = sum(1 for _, _, success in results if success)
  total_slices = sum(num_slices for _, num_slices, _ in results)

  print(f"\nCompleted {split}:")
  print(f"  - Volumes processed: {successful}/{len(volume_files)}")
  print(f"  - Total slices: {total_slices}")

  # Create slice index
  create_slice_index(output_base, split)

  return successful, total_slices


def main():
  parser = argparse.ArgumentParser(
    description="Preprocess GDPHMM volumes into individual slices"
  )
  parser.add_argument(
    '--dataset',
    type=str,
    default='gdphmm-hn',
    help='Dataset name (default: gdphmm-hn)'
  )
  parser.add_argument(
    '--splits',
    nargs='+',
    default=['train', 'valid', 'test'],
    help='Splits to process (default: train valid test)'
  )
  parser.add_argument(
    '--voxel-spacing',
    nargs=3,
    type=float,
    default=[2.5, 2.5, 2.0],
    help='Voxel spacing in mm (sx sy sz) for SDM computation (default: [2.5, 2.5, 2.0])'
  )
  parser.add_argument(
    '--pad-size',
    nargs=2,
    type=int,
    default=[176, 160],
    help='Pad slices to fixed size (H W). If not specified, slices stored at original size. Example: --pad-size 256 256'
  )
  parser.add_argument(
    '--workers',
    type=int,
    default=cpu_count(),
    help=f'Number of parallel workers (default: {cpu_count()})'
  )
  parser.add_argument(
    '--limit',
    type=int,
    default=None,
    help='Limit number of volumes per split (for testing)'
  )

  args = parser.parse_args()

  print("="*80)
  print("GDPHMM Slice Preprocessing")
  print("="*80)
  print(f"Dataset: {args.dataset}")
  print(f"Splits: {args.splits}")
  print(f"Voxel spacing: {args.voxel_spacing}")
  print(f"Pad size: {args.pad_size if args.pad_size else 'None (original size)'}")
  print(f"Workers: {args.workers}")
  if args.limit:
    print(f"Limit: {args.limit} volumes per split")
  print("="*80)

  voxel_spacing = tuple(args.voxel_spacing)
  pad_size = tuple(args.pad_size) if args.pad_size else None

  total_volumes = 0
  total_slices = 0

  for split in args.splits:
    try:
      num_volumes, num_slices = preprocess_split(
        args.dataset,
        split,
        voxel_spacing,
        args.workers,
        pad_size,
        args.limit
      )
      total_volumes += num_volumes
      total_slices += num_slices
    except Exception as e:
      print(f"Error processing split {split}: {e}")
      import traceback
      traceback.print_exc()

  print("\n" + "="*80)
  print("PREPROCESSING COMPLETE")
  print("="*80)
  print(f"Total volumes processed: {total_volumes}")
  print(f"Total slices created: {total_slices}")
  print(f"\nOutput directory: {GDPHMM_DATA_BASE_PATH / f'{args.dataset}_slices'}")
  print("="*80)


if __name__ == '__main__':
  main()
