import sys
from pathlib import Path
# sys.path.append(str(Path().resolve().parent / 'src'))

import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import read_model_config
from trainer import Trainer


def train_worker(rank, world_size, config, n_epochs):
  try:
    setup_distributed(rank, world_size)

    trainer = Trainer(config, n_epochs=n_epochs, rank=rank, world_size=world_size)
    trainer.train()
  except Exception as e:
    print(f"Error in worker {rank}: {e}")
    raise
  finally:
    cleanup_distributed()


def setup_distributed(rank, world_size, backend='nccl'):
  """Initialize distributed training."""
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '12355'

  # Initialize the process group
  dist.init_process_group(backend, rank=rank, world_size=world_size)


def cleanup_distributed():
  """Clean up distributed training."""
  dist.destroy_process_group()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train the model. If None, use the config value.')
  parser.add_argument('--distributed', action='store_true', help='Set to True to use distributed training across multiple GPUs.')
  args = parser.parse_args()

  print('Training with arguments', args)
  print()

  config = read_model_config(args.config)
  print('Configuration')
  print(json.dumps(config, indent=2))
  print()

  mp.set_start_method('spawn', force=True)

  if args.distributed:
    world_size = torch.cuda.device_count()
    if world_size == 0:
      print("No CUDA devices available. Falling back to single-process training.")
      world_size = 1

    print(f'Training with {world_size} process(es)')
    print()

    mp.spawn(
      train_worker,
      args=(world_size, config, args.epochs),
      nprocs=world_size,
      join=True
    )
  else:
    print(f'Training with a single process')
    print()

    trainer = Trainer(config, n_epochs=args.epochs)
    trainer.train()

  print('Training completed successfully.')

