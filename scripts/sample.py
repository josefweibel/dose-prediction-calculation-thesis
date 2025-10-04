import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import sys
import atexit

from config import read_sampling_config
from sampler import Sampler


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--config', type=str, required=True, help='Name of the sampling configuration file (without .yaml extension)')
  parser.add_argument('--data-split', type=str, choices=['train', 'val', 'test'], default='test', help='Data split to use for sampling.')
  parser.add_argument('--epoch', type=int, default=None, help='Use checkpoint from this epoch. If None, use the latest checkpoint.')
  parser.add_argument('--distributed', action='store_true', help='Set to True to use distributed sampling across multiple GPUs.')
  args = parser.parse_args()

  print('Sampling with arguments', args)
  print()

  sampling_config = read_sampling_config(args.config)
  print('Configuration')
  print(json.dumps(sampling_config, indent=2))
  print()

  mp.set_start_method('spawn', force=True)

  if args.distributed:
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "12355")

    print(f"Distributed sampling: rank {rank}, local_rank {local_rank}, world_size {world_size}", flush=True)
    print(f"{rank} tcp://{master_addr}:{master_port}", flush=True)

    torch.cuda.set_device(local_rank)

    dist.init_process_group(
      backend="nccl",
      init_method=f"tcp://{master_addr}:{master_port}",
      rank=rank,
      world_size=world_size,
      device_id=local_rank,
    )

    dist.barrier()

    try:
      sampler = Sampler(
        model_name=args.model,
        sampling_config=sampling_config,
        data_split=args.data_split,
        epoch=args.epoch if args.epoch and len(args.epoch) else None,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
      )

      def cleanup():
        print('cleanup of sampler tried')
        sampler.finish()

      atexit.register(cleanup)

      def handle_signal(signum, frame):
        print(f"Received signal {signum}")
        cleanup()
        sys.exit(0)

      signal.signal(signal.SIGTERM, handle_signal)
      signal.signal(signal.SIGINT, handle_signal)

      try:
        sampler.train()
      finally:
        sampler.finish()
    finally:
      dist.destroy_process_group()
  else:
    print(f'Sampling with a single process')
    print()

    sampler = Sampler(
      model_name=args.model,
      sampling_config=sampling_config,
      data_split=args.data_split,
      epoch=args.epochs,
    )

    def cleanup():
      print('cleanup of sampler tried')
      sampler.finish()

    atexit.register(cleanup)

    def handle_signal(signum, frame):
      print(f"Received signal {signum}")
      cleanup()
      sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    sampler.train()

  print('Sampling script completed.')

