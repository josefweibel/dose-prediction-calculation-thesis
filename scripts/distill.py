"""
Progressive Distillation Training Script.

Distills a trained diffusion model to use fewer sampling steps while
maintaining quality. Implements progressive distillation (Salimans & Ho, 2022).

Usage:
  # Full distillation (1024 -> 512 -> ... -> 4 steps)
  python scripts/distill.py --model 101-beamdiff --distill progressive-v1

  # Resume from a specific round
  python scripts/distill.py --model 101-beamdiff --distill progressive-v1 --start-round 3

  # Distributed training
  python scripts/distill.py --model 101-beamdiff --distill progressive-v1 --distributed
"""

import argparse
import atexit
import json
import os
import signal
import sys
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from config import read_model_config, read_distillation_config
from distillation import ProgressiveDistillationOrchestrator
from utils.mp import detect_device


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Progressive Distillation for Diffusion Models')
  parser.add_argument('--model', type=str, required=True,
            help='Model config name (e.g., 101-beamdiff)')
  parser.add_argument('--distill', type=str, required=True,
            help='Distillation config name (e.g., progressive-v1)')
  parser.add_argument('--start-round', type=int, default=0,
            help='Round to start from (0-based, default: 0)')
  parser.add_argument('--end-round', type=int, default=None,
            help='Round to end at (exclusive, default: all)')
  parser.add_argument('--distributed', action='store_true',
            help='Use distributed training across multiple GPUs')
  args = parser.parse_args()

  print('Distillation with arguments', args)
  print()

  # Load configurations
  model_config = read_model_config(args.model)
  distill_config = read_distillation_config(args.distill)

  print('Model Configuration')
  print(json.dumps(model_config, indent=2))
  print()

  print('Distillation Configuration')
  print(json.dumps(distill_config, indent=2))
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

    print(f"Distributed distillation: rank {rank}, local_rank {local_rank}, world_size {world_size}", flush=True)
    print(f"{rank} tcp://{master_addr}:{master_port}", flush=True)

    # With --gpus-per-task, SLURM sets CUDA_VISIBLE_DEVICES for each task
    # Each task sees only its assigned GPU(s), so we use device 0
    torch.cuda.set_device(0)
    device = torch.device('cuda', 0)

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
      orchestrator = ProgressiveDistillationOrchestrator(
        model_config=model_config,
        distill_config=distill_config,
        device=device,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
      )

      def cleanup():
        print('Cleanup attempted')

      atexit.register(cleanup)

      def handle_signal(signum, frame):
        print(f"Received signal {signum}")
        cleanup()
        sys.exit(0)

      signal.signal(signal.SIGTERM, handle_signal)
      signal.signal(signal.SIGINT, handle_signal)

      orchestrator.run_all(
        start_round=args.start_round,
        end_round=args.end_round,
      )
    finally:
      dist.destroy_process_group()
  else:
    print('Distillation with a single process')
    print()

    device = detect_device()

    orchestrator = ProgressiveDistillationOrchestrator(
      model_config=model_config,
      distill_config=distill_config,
      device=device,
    )

    def cleanup():
      print('Cleanup attempted')

    atexit.register(cleanup)

    def handle_signal(signum, frame):
      print(f"Received signal {signum}")
      cleanup()
      sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    orchestrator.run_all(
      start_round=args.start_round,
      end_round=args.end_round,
    )

  print('Distillation script completed.')
