import argparse
import json
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm
import signal
import sys
import atexit


def combine_slices_to_volumes(model_name, sampling_name, data_split, epoch, delete_slices=False):
  """Combine individual slice .npy files into volume .npy files.

  Args:
    model_name: Name of the model
    sampling_name: Name of the sampling configuration
    data_split: Data split (train/val/test)
    epoch: Epoch number
    delete_slices: Whether to delete individual slice files after combining
  """

  # Construct the path to samples
  base_path = Path(__file__).parents[1]
  samples_path = base_path / 'samples' / model_name / sampling_name / f'{data_split}_{epoch}'

  if not samples_path.exists():
    print(f'Error: Samples path does not exist: {samples_path}')
    sys.exit(1)

  print(f'Combining slices in: {samples_path}')
  print()

  # Find all slice files (format: {sample_name}-{slice_index}.npy)
  slice_files = list(samples_path.glob('*-*.npy'))

  if len(slice_files) == 0:
    print('No slice files found to combine.')
    return

  print(f'Found {len(slice_files)} slice files')

  # Group slice files by sample name
  samples_dict = {}
  for slice_file in slice_files:
    # Parse filename: {sample_name}-{slice_index}.npy
    parts = slice_file.stem.rsplit('-', 1)
    if len(parts) != 2:
      print(f'Warning: Skipping file with unexpected format: {slice_file.name}')
      continue

    sample_name, slice_index_str = parts
    try:
      slice_index = int(slice_index_str)
    except ValueError:
      print(f'Warning: Skipping file with non-integer slice index: {slice_file.name}')
      continue

    if sample_name not in samples_dict:
      samples_dict[sample_name] = []

    samples_dict[sample_name].append((slice_index, slice_file))

  print(f'Found {len(samples_dict)} unique samples')
  print()

  # Process each sample
  for sample_name, slices in tqdm(samples_dict.items(), desc='Combining samples'):
    # Sort by slice index
    slices.sort(key=lambda x: x[0])

    # Determine the number of slices
    slice_indices = [idx for idx, _ in slices]
    max_slice_index = max(slice_indices)
    num_slices = max_slice_index + 1  # Assuming 0-indexed

    # Load the first slice to get the shape
    first_slice_data = np.load(slices[0][1])
    slice_shape = first_slice_data.shape

    # Initialize the volume array
    # Shape: (channels, height, width, num_slices)
    volume_shape = (*slice_shape, num_slices)
    volume = np.zeros(volume_shape, dtype=first_slice_data.dtype)

    # Load each slice and place it in the volume
    for slice_index, slice_file in slices:
      slice_data = np.load(slice_file)
      volume[..., slice_index] = slice_data

    # Save the combined volume
    output_path = samples_path / f'{sample_name}.npy'
    np.save(output_path, volume)

  print()
  print(f'Successfully combined {len(samples_dict)} samples into volumes')
  print(f'Output location: {samples_path}')

  # Delete old slice files if requested
  if delete_slices:
    print()
    print('Deleting individual slice files...')
    deleted_count = 0
    for sample_name, slices in tqdm(samples_dict.items(), desc='Deleting slices'):
      for slice_index, slice_file in slices:
        try:
          slice_file.unlink()
          deleted_count += 1
        except Exception as e:
          print(f'Warning: Failed to delete {slice_file.name}: {e}')

    print(f'Deleted {deleted_count} slice files')
    print('Cleanup completed')
  else:
    print('Individual slice files kept (use --delete-slices to remove them)')


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Combine individual slice files into volume files')
  parser.add_argument('--model', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--sampling', type=str, required=True, help='Name of the sampling configuration file (without .yaml extension)')
  parser.add_argument('--data-split', type=str, choices=['train', 'val', 'test'], default='test', help='Data split to use for sampling.')
  parser.add_argument('--epoch', type=int, required=True, help='Epoch number to use.')
  parser.add_argument('--delete-slices', action='store_true', help='Delete individual slice files after combining into volumes.')
  args = parser.parse_args()

  print('Combining slices with arguments', args)
  print()

  combine_slices_to_volumes(
    model_name=args.model,
    sampling_name=args.sampling,
    data_split=args.data_split,
    epoch=args.epoch,
    delete_slices=args.delete_slices,
  )

  print('Slice combining script completed.')
