#!/usr/bin/env python3
"""
Preprocessing script for GDPHMMSlicesDataset.

This script processes 3D volumes and saves them as individual 2D slices,
ready to be loaded directly without on-the-fly preprocessing.

All transformations from the original dataset are applied:
  - CT: clipped to [-1024, 3071], normalized to [0, 1], stored as float32
  - Dose: normalized to [0, 1], stored as float32
  - PTV: 3 masks sorted by dose descending (highest first), stored as uint8 with dose values
  - Body: stored as uint8
  - OAR: binary masks, stored as uint8 (filtered per patient using Pat_Obj_DICT.json)
  - OAR SDM: clipped to [0, 40] mm, normalized to [0, 1], stored as float16
  - PTV SDM: clipped to [-50, 50] mm, normalized to [-1, 1], stored as float16
  - Beam plate: clipped to [0, 92], normalized to [0, 1], stored as float32

IMPORTANT: SDMs are computed AFTER padding to ensure correct distance values at edges.

PTV EXTRACTION: Uses PTV_DICT.json to get patient-specific PTV configurations.
Each patient may have 1-3 PTVs (PTV_High, PTV_Mid, PTV_Low) with different prescribed doses.
PTVs are sorted by dose descending and stored in a (3, H, W) tensor. Missing PTVs are zeros.

OAR FILTERING: Uses Pat_Obj_DICT.json to determine which OARs are available per patient.

Output structure:
  data/{dataset_name}_slices/{split}/{volume_name}/slice_{idx:04d}.npz

Each slice contains:
  - ct: (1, H, W) float32 - clipped and normalized CT scan
  - dose: (1, H, W) float32 - normalized dose distribution
  - ptv_doses: (3, H, W) uint8 - PTV masks sorted by dose descending, values are prescribed dose (e.g., 70, 63, 56)
  - ptv_sdm: (3, H, W) float16 - signed distance maps for each PTV
  - body: (1, H, W) uint8 - body mask
  - oar: (N_OAR, H, W) uint8 - binary OAR masks
  - oar_sdm: (N_OAR, H, W) float16 - signed distance maps for OARs
  - beam_plate: (1, H, W) float32 - beam configuration plate
  - angle_plate: (1, H, W) uint8 - angle configuration plate (binary, 3D around isocenter)
"""

import numpy as np
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import argparse
from scipy.ndimage import distance_transform_edt, zoom
from multiprocessing import Pool, cpu_count
import sys
import json

# Import constants from data module
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from data import (
  GDPHMM_DATA_BASE_PATH,
  GDPHMM_IMG_MIN,
  GDPHMM_IMG_MAX,
  GDPHMM_DOSE_MIN,
  GDPHMM_DOSE_MAX,
  GDPHMM_OAR_SDM_MIN,
  GDPHMM_OAR_SDM_MAX,
  GDPHMM_PTV_SDM_MIN,
  GDPHMM_PTV_SDM_MAX,
  GDPHMM_BEAM_PLATE_MIN,
  GDPHMM_BEAM_PLATE_MAX,
  GDPHMM_OAR,
)

def process_angle_plate(angle_plate_2d, img_shape, isocenter, z_range=5):
  """
  Convert 2D angle plate to 3D volume around the isocenter.

  Based on GDP-HMM implementation: creates a 3D volume by replicating the 2D binary angle plate
  within a range around the isocenter (±z_range slices).

  Args:
    angle_plate_2d: (H, W) 2D binary angle plate array
    img_shape: (H, W, D) shape of the target volume
    isocenter: (z, y, x) coordinates of the isocenter
    z_range: number of slices above and below isocenter to fill (default: 5)

  Returns:
    angle_plate_3d: (H, W, D) 3D binary volume with angle plate around isocenter
  """
  # Initialize empty 3D volume
  angle_plate_3d = np.zeros(img_shape, dtype=np.uint8)

  # Determine z-range around isocenter
  z_begin = max(0, int(isocenter[0]) - z_range)
  z_end = min(img_shape[2], int(isocenter[0]) + z_range)

  # Resize angle_plate to match img_shape if needed (order=0 preserves binary values)
  if angle_plate_2d.shape[0] != img_shape[0] or angle_plate_2d.shape[1] != img_shape[1]:
    zoom_factors = (img_shape[0] / angle_plate_2d.shape[0],
                    img_shape[1] / angle_plate_2d.shape[1])
    angle_plate_2d = zoom(angle_plate_2d, zoom_factors, order=0)

  # Replicate the 2D plate across the z-range
  for z in range(z_begin, z_end):
    angle_plate_3d[:, :, z] = angle_plate_2d

  return angle_plate_3d


def load_pat_obj_dict():
  """
  Load the Pat_Obj_DICT.json file that maps patient IDs to available OARs.

  Returns:
    Dictionary mapping patient IDs to lists of OAR names, or None if file not found
  """
  pat_obj_path = GDPHMM_DATA_BASE_PATH / 'gdphmm-hn' / 'incoming' / 'Pat_Obj_DICT.json'

  if not pat_obj_path.exists():
    print(f"Warning: Pat_Obj_DICT.json not found at {pat_obj_path}")
    return None

  with open(pat_obj_path, 'r') as f:
    pat_obj_dict = json.load(f)

  print(f"Loaded Pat_Obj_DICT.json with {len(pat_obj_dict)} patients")
  return pat_obj_dict


def load_ptv_dict():
  """Load PTV_DICT.json mapping patient IDs to PTV configurations."""
  ptv_path = GDPHMM_DATA_BASE_PATH / 'gdphmm-hn' / 'incoming' / 'PTV_DICT.json'
  with open(ptv_path, 'r') as f:
    ptv_dict = json.load(f)
  print(f"Loaded PTV_DICT.json with {len(ptv_dict)} patients")
  return ptv_dict


def combine_ptv(vol, ptv_config):
  """
  Combine PTV masks similar to GDP-HMM's combine_ptv.

  Returns:
    cat_ptv: (3, H, W, D) - 3 masks sorted by dose descending (highest first).
             Each mask contains the prescribed dose value (e.g., 70, 63, 56 Gy).
             Missing PTVs are zeros.
  """
  shape = vol['img'].shape

  # Collect available PTVs with their doses
  ptv_list = []
  for level in ['PTV_High', 'PTV_Mid', 'PTV_Low']:
    if level in ptv_config:
      opt_name = ptv_config[level]['OPTName']
      pdose = ptv_config[level]['PDose']
      ptv_list.append((vol[opt_name].astype(np.float32), pdose))

  # Sort by dose descending
  ptv_list.sort(key=lambda x: x[1], reverse=True)

  # Pad to exactly 3 masks
  while len(ptv_list) < 3:
    ptv_list.append((np.zeros(shape, dtype=np.float32), 0.0))

  # Stack masks with prescribed dose value
  cat_ptv = np.stack([mask * pdose for mask, pdose in ptv_list], axis=0)

  return cat_ptv


def compute_sdm(mask: np.ndarray, voxel_spacing: tuple = (1.0, 1.0, 1.0)) -> np.ndarray:
  """
  Compute Signed Distance Map (SDM) for a binary mask.

  SDM provides the minimum distance from each voxel to the contour of the ROI,
  where positive values are inside the ROI, negative values are outside, and
  zero values are on the boundary.

  Args:
      mask: Binary mask (H, W) or (H, W, D) where 1=inside ROI, 0=outside
      voxel_spacing: Physical spacing (is, js) or (is, js, ks) in mm

  Returns:
      SDM with positive values inside, negative outside, 0 on boundary
  """
  mask = mask.astype(bool)

  # Distance transform for points inside (positive distances)
  dist_inside = distance_transform_edt(mask, sampling=voxel_spacing)

  # Distance transform for points outside (will be negated)
  dist_outside = distance_transform_edt(~mask, sampling=voxel_spacing)

  # Combine: positive inside, negative outside
  sdm = dist_inside - dist_outside

  return sdm


def compute_sdms(oar_vol, ptv_doses, voxel_spacing):
  """
  Compute signed distance maps for OAR and PTV masks.

  OAR SDMs are clipped to [GDPHMM_OAR_SDM_MIN, GDPHMM_OAR_SDM_MAX] mm and normalized to [0, 1].
  PTV SDMs are clipped to [GDPHMM_PTV_SDM_MIN, GDPHMM_PTV_SDM_MAX] mm and normalized to [-1, 1].

  Args:
    oar_vol: (N_OAR, H, W, D) - binary masks for organs at risk
    ptv_doses: (3, H, W, D) - PTV masks sorted by dose (highest first), values are dose/GDPHMM_DOSE_MAX
    voxel_spacing: tuple of (sx, sy, sz) - voxel spacing in mm

  Returns:
    oar_sdm: (N_OAR, H, W, D) float32 - clipped and normalized SDMs for each OAR [0, 1]
    ptv_sdm: (3, H, W, D) float32 - clipped and normalized SDMs for each PTV [-1, 1]
  """
  # Compute raw OAR SDMs
  oar_sdm = np.stack([
    compute_sdm(oar_vol[i], voxel_spacing)
    for i in range(oar_vol.shape[0])
  ], axis=0)

  # Clip OAR SDMs and normalize to [0, 1]
  oar_sdm = np.clip(oar_sdm, GDPHMM_OAR_SDM_MIN, GDPHMM_OAR_SDM_MAX)
  oar_sdm = (oar_sdm - GDPHMM_OAR_SDM_MIN) / (GDPHMM_OAR_SDM_MAX - GDPHMM_OAR_SDM_MIN)

  # Compute SDMs for each PTV mask (binary: >0)
  ptv_sdm = np.stack([
    compute_sdm((ptv_doses[i] > 0).astype(np.uint8), voxel_spacing)
    for i in range(3)
  ], axis=0)

  # Clip PTV SDMs and normalize to [-1, 1]
  ptv_sdm = np.clip(ptv_sdm, GDPHMM_PTV_SDM_MIN, GDPHMM_PTV_SDM_MAX)
  ptv_sdm = (ptv_sdm - GDPHMM_PTV_SDM_MIN) / (GDPHMM_PTV_SDM_MAX - GDPHMM_PTV_SDM_MIN) * 2 - 1

  return oar_sdm, ptv_sdm


def apply_transformations(ct_vol, dose_vol, ptv_doses, body_vol, oar_vol, beam_plate_vol, angle_plate_vol, pad_size=None):
  """
  Apply all transformations to match the GDPHMMSlicesDataset behavior.

  This includes:
    - CT: clip to [-1024, 3071], normalize to [0, 1]
    - Dose: normalize to [0, 1]
    - PTV masks: store dose levels as uint8 (e.g., 0, 56, 63, 70)
    - Body: convert to uint8
    - OAR: convert to uint8
    - Beam plate: clip to [0, 92], normalize to [0, 1]
    - Angle plate: convert to uint8 (binary)
    - Padding: optionally pad to fixed size (saves runtime padding)

  Args:
    ct_vol: (H, W, D) float - CT scan
    dose_vol: (H, W, D) float - dose distribution
    ptv_doses: (3, H, W, D) float - PTV masks with prescribed dose values
    body_vol: (H, W, D) - body mask
    oar_vol: (N_OAR, H, W, D) - OAR binary masks
    beam_plate_vol: (H, W, D) - beam configuration plate
    angle_plate_vol: (H, W, D) - angle configuration plate
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

  # PTV masks: store as uint8 (dose values like 0, 56, 63, 70)
  ptv_doses_transformed = np.round(ptv_doses).astype(np.uint8)

  # Body: convert to uint8
  body_transformed = body_vol.astype(np.uint8)

  # OAR: convert to uint8
  oar_transformed = oar_vol.astype(np.uint8)

  # Beam plate: clip and normalize to [0, 1]
  beam_plate_transformed = np.clip(beam_plate_vol, GDPHMM_BEAM_PLATE_MIN, GDPHMM_BEAM_PLATE_MAX)
  beam_plate_transformed = (beam_plate_transformed - GDPHMM_BEAM_PLATE_MIN) / (GDPHMM_BEAM_PLATE_MAX - GDPHMM_BEAM_PLATE_MIN)
  beam_plate_transformed = beam_plate_transformed.astype(np.float32)

  # Angle plate: convert to uint8 (binary mask)
  angle_plate_transformed = angle_plate_vol.astype(np.uint8)

  # Optionally pad all volumes to fixed size using np.pad
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
    pad_2d = ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
    pad_3d = ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right), (0, 0))

    ct_transformed = np.pad(ct_transformed, pad_2d, mode='constant', constant_values=0)
    dose_transformed = np.pad(dose_transformed, pad_2d, mode='constant', constant_values=0)
    ptv_doses_transformed = np.pad(ptv_doses_transformed, pad_3d, mode='constant', constant_values=0)
    body_transformed = np.pad(body_transformed, pad_2d, mode='constant', constant_values=0)
    oar_transformed = np.pad(oar_transformed, pad_3d, mode='constant', constant_values=0)
    beam_plate_transformed = np.pad(beam_plate_transformed, pad_2d, mode='constant', constant_values=0)
    angle_plate_transformed = np.pad(angle_plate_transformed, pad_2d, mode='constant', constant_values=0)

  return ct_transformed, dose_transformed, ptv_doses_transformed, body_transformed, oar_transformed, beam_plate_transformed, angle_plate_transformed


def process_volume(args):
  """
  Process a single volume file and save all its slices.

  Args:
    args: tuple of (file_path, output_dir, voxel_spacing, pad_size, pat_obj_dict, ptv_dict)

  Returns:
    tuple of (volume_name, num_slices, success)
  """
  file_path, output_dir, voxel_spacing, pad_size, pat_obj_dict, ptv_dict = args

  try:
    # Load volume
    vol = np.load(file_path, allow_pickle=True)['arr_0'].item()

    # Extract base data
    dose_vol = (vol['dose'] * vol['dose_scale'] * vol['Body'])
    ct_vol = vol['img']
    body_vol = vol['Body']

    # Extract patient ID from filename (e.g., '0522c0001+15Ag+MOS_23629.npz' -> '0522c0001')
    patient_id = file_path.stem.split('+')[0]

    # Get list of available OARs for this patient
    if pat_obj_dict is not None and patient_id in pat_obj_dict:
      available_oars = set(pat_obj_dict[patient_id])
    else:
      available_oars = None  # All OARs considered available

    # Build OAR volume (N_OAR, H, W, D)
    oar_vol = np.stack(
      [vol[name].astype(np.uint8) if (name in vol and (available_oars is None or name in available_oars))
       else np.zeros_like(vol['dose'].astype(np.uint8))
       for name in GDPHMM_OAR],
      axis=0
    )

    # Build PTV masks using patient-specific config
    ptv_config = ptv_dict[patient_id]
    ptv_doses = combine_ptv(vol, ptv_config)

    # Extract beam_plate (3D), angle_plate (2D), and isocenter
    beam_plate_vol = vol['beam_plate']
    angle_plate_2d = vol['angle_plate']
    isocenter = vol['isocenter']

    # Process angle_plate from 2D to 3D volume around isocenter
    angle_plate_vol = process_angle_plate(angle_plate_2d, ct_vol.shape, isocenter, z_range=5)

    # Apply all transformations (clipping, normalization, type conversion, padding)
    # NOTE: Padding is applied BEFORE SDM computation to avoid 0-valued edges in distance maps
    ct_vol, dose_vol, ptv_doses, body_vol, oar_vol, beam_plate_vol, angle_plate_vol = apply_transformations(
      ct_vol, dose_vol, ptv_doses, body_vol, oar_vol, beam_plate_vol, angle_plate_vol, pad_size=pad_size
    )

    # Compute signed distance maps AFTER padding so distances are correct at edges
    oar_sdm, ptv_sdm = compute_sdms(oar_vol, ptv_doses, voxel_spacing)

    # Convert SDMs to float16 for storage efficiency
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
        'ptv_doses': ptv_doses[:, :, :, slice_idx],  # (3, H, W)
        'ptv_sdm': ptv_sdm[:, :, :, slice_idx],  # (3, H, W)
        'body': body_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'oar': oar_vol[:, :, :, slice_idx],  # (N_OAR, H, W)
        'oar_sdm': oar_sdm[:, :, :, slice_idx],  # (N_OAR, H, W)
        'beam_plate': beam_plate_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
        'angle_plate': angle_plate_vol[:, :, slice_idx][np.newaxis, :, :],  # (1, H, W)
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

  # Load config dictionaries
  pat_obj_dict = load_pat_obj_dict()
  ptv_dict = load_ptv_dict()

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
  args_list = [(f, output_dir, voxel_spacing, pad_size, pat_obj_dict, ptv_dict) for f in volume_files]

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
