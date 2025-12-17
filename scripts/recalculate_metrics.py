import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import sys
import atexit
from datetime import timedelta

from metrics_recalculator import MetricsRecalculator


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--sampling', type=str, required=True, help='Name of the sampling configuration file (without .yaml extension)')
  parser.add_argument('--data-split', type=str, choices=['train', 'val', 'test'], default='test', help='Data split to use for sampling.')
  parser.add_argument('--epoch', type=int, default=None, help='Use checkpoint from this epoch.')
  parser.add_argument('--distributed', action='store_true', help='Set to True to use distributed computation across multiple GPUs.')
  args = parser.parse_args()

  print('Metric Recalculation with arguments', args)
  print()

  mp.set_start_method('spawn', force=True)

  if args.distributed:
    print("CUDA_VISIBLE_DEVICES =", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("torch.cuda.device_count() =", torch.cuda.device_count())

    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["SLURM_NTASKS"])
    master_addr = os.environ["MASTER_ADDR"]
    master_port = os.environ.get("MASTER_PORT", "12355")

    print(f"Distributed metric recalculation: rank {rank}, local_rank {local_rank}, world_size {world_size}", flush=True)
    print(f"{rank} tcp://{master_addr}:{master_port}", flush=True)

    # With --gpus-per-task, SLURM sets CUDA_VISIBLE_DEVICES for each task
    # Each task sees only its assigned GPU(s), so we use device 0
    torch.cuda.set_device(0)

    # Set a longer timeout for NCCL operations (30 minutes instead of default 10 minutes)
    # This is needed because volume metrics computation can take a long time
    dist.init_process_group(
      backend="nccl",
      init_method=f"tcp://{master_addr}:{master_port}",
      rank=rank,
      world_size=world_size,
      device_id=0,
      timeout=timedelta(minutes=30),
    )

    dist.barrier()

    try:
      metric_recalculator = MetricsRecalculator(
        model_name=args.model,
        sampling_name=args.sampling,
        data_split=args.data_split,
        epoch=args.epoch,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
      )

      def cleanup():
        print('cleanup of metric recalculator tried')
        metric_recalculator.finish()

      atexit.register(cleanup)

      def handle_signal(signum, frame):
        print(f"Received signal {signum}")
        cleanup()
        sys.exit(0)

      signal.signal(signal.SIGTERM, handle_signal)
      signal.signal(signal.SIGINT, handle_signal)

      try:
        metric_recalculator.calculate()
      finally:
        metric_recalculator.finish()
    finally:
      dist.destroy_process_group()
  else:
    print(f'Metric recalculation with a single process')
    print()

    metric_recalculator = MetricsRecalculator(
      model_name=args.model,
      sampling_name=args.sampling,
      data_split=args.data_split,
      epoch=args.epoch,
    )

    def cleanup():
      print('cleanup of metric recalculator tried')
      metric_recalculator.finish()

    atexit.register(cleanup)

    def handle_signal(signum, frame):
      print(f"Received signal {signum}")
      cleanup()
      sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    metric_recalculator.calculate()

  print('Metric recalculation script completed.')

