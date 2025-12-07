import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys


class SliceCombinator:
  def __init__(
    self,
    model_name,
    sampling_name,
    data_split,
    epoch=None,
    delete_slices=False,
    rank=None,
    world_size=None,
  ):
    self.model_name = model_name
    self.sampling_name = sampling_name
    self.data_split = data_split
    self.epoch = epoch
    self.delete_slices = delete_slices
    self.rank = rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    # Construct the path to samples
    base_path = Path(__file__).parents[1]

    # Handle "latest" epoch
    if epoch == "latest" or epoch is None:
      sampling_base = base_path / 'samples' / model_name / sampling_name
      epoch_dirs = sorted(sampling_base.glob(f'{data_split}_*'))
      if not epoch_dirs:
        if self.is_main_process:
          print(f'Error: No epoch directories found in {sampling_base}')
        sys.exit(1)

      samples_path = epoch_dirs[-1]  # Get the last one (highest epoch number)
      self.epoch = int(samples_path.name.split('_')[-1])
      if self.is_main_process:
        print(f'Using latest epoch: {self.epoch}')
    else:
      samples_path = base_path / 'samples' / model_name / sampling_name / f'{data_split}_{epoch}'

    if not samples_path.exists():
      if self.is_main_process:
        print(f'Error: Samples path does not exist: {samples_path}')
      sys.exit(1)

    self.samples_path = samples_path

    if self.is_main_process:
      print(f'Combining slices in: {self.samples_path}')
      print()

  def combine(self):
    """Combine individual slice .npy files into volume .npy files."""
    # Find all slice files (format: {sample_name}-{slice_index}.npy)
    slice_files = list(self.samples_path.glob('*-*.npy'))

    if len(slice_files) == 0:
      if self.is_main_process:
        print('No slice files found to combine.')
      return

    if self.is_main_process:
      print(f'Found {len(slice_files)} slice files')

    # Group slice files by sample name
    samples_dict = {}
    for slice_file in slice_files:
      # Parse filename: {sample_name}-{slice_index}.npy
      parts = slice_file.stem.rsplit('-', 1)
      if len(parts) != 2:
        if self.is_main_process:
          print(f'Warning: Skipping file with unexpected format: {slice_file.name}')
        continue

      sample_name, slice_index_str = parts
      try:
        slice_index = int(slice_index_str)
      except ValueError:
        if self.is_main_process:
          print(f'Warning: Skipping file with non-integer slice index: {slice_file.name}')
        continue

      if sample_name not in samples_dict:
        samples_dict[sample_name] = []

      samples_dict[sample_name].append((slice_index, slice_file))

    if self.is_main_process:
      print(f'Found {len(samples_dict)} unique samples')
      print()

    # Process each sample
    pbar = tqdm(samples_dict.items(), desc='Combining samples', disable=not self.is_main_process)
    for sample_name, slices in pbar:
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
      output_path = self.samples_path / f'{sample_name}.npy'
      np.save(output_path, volume)

    if self.is_main_process:
      print()
      print(f'Successfully combined {len(samples_dict)} samples into volumes')
      print(f'Output location: {self.samples_path}')

    # Delete old slice files if requested
    if self.delete_slices:
      if self.is_main_process:
        print()
        print('Deleting individual slice files...')

      deleted_count = 0
      pbar = tqdm(samples_dict.items(), desc='Deleting slices', disable=not self.is_main_process)
      for sample_name, slices in pbar:
        for slice_index, slice_file in slices:
          try:
            slice_file.unlink()
            deleted_count += 1
          except Exception as e:
            if self.is_main_process:
              print(f'Warning: Failed to delete {slice_file.name}: {e}')

      if self.is_main_process:
        print(f'Deleted {deleted_count} slice files')
        print('Cleanup completed')
    else:
      if self.is_main_process:
        print('Individual slice files kept (use --delete-slices to remove them)')

  def finish(self):
    """Cleanup method for consistency with Trainer and Sampler."""
    if self.is_main_process:
      print('finishing slice combinator instance', flush=True)
