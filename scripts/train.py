import sys
from pathlib import Path
# sys.path.append(str(Path().resolve().parent / 'src'))

import argparse
import json
import os
import signal
import sys
import atexit
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import read_model_config
from trainer import Trainer


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
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "12355")

    print(f"Distributed training: rank {rank}, local_rank {local_rank}, world_size {world_size}", flush=True)
    print(f"{rank} tcp://{master_addr}:{master_port}", flush=True)

    # With --gpus-per-task, SLURM sets CUDA_VISIBLE_DEVICES for each task
    # Each task sees only its assigned GPU(s), so we use device 0
    torch.cuda.set_device(0)

    dist.init_process_group(
      backend="nccl",
      init_method=f"tcp://{master_addr}:{master_port}",
      rank=rank,
      world_size=world_size,
      device_id=0,
    )

    dist.barrier()

    try:
      trainer = Trainer(config, n_epochs=args.epochs, rank=rank, world_size=world_size, local_rank=local_rank)

      def cleanup():
        print('cleanup of trainer tried')
        trainer.finish()

      atexit.register(cleanup)

      def handle_signal(signum, frame):
        print(f"Received signal {signum}")
        cleanup()
        sys.exit(0)

      signal.signal(signal.SIGTERM, handle_signal)
      signal.signal(signal.SIGINT, handle_signal)

      try:
        trainer.train()
      finally:
        trainer.finish()
    finally:
      dist.destroy_process_group()
  else:
    print(f'Training with a single process')
    print()

    trainer = Trainer(config, n_epochs=args.epochs)

    def cleanup():
      print('cleanup of trainer tried')
      trainer.finish()

    atexit.register(cleanup)

    def handle_signal(signum, frame):
      print(f"Received signal {signum}")
      cleanup()
      sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    trainer.train()

  print('Training script completed.')

