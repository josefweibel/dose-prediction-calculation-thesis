import argparse
import json
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import signal
import sys
import atexit

from metrics_recalculator import MetricsRecalculator


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--sampling', type=str, required=True, help='Name of the sampling configuration file (without .yaml extension)')
  parser.add_argument('--data-split', type=str, choices=['train', 'val', 'test'], default='test', help='Data split to use for sampling.')
  parser.add_argument('--epoch', type=int, default=None, help='Use checkpoint from this epoch.')
  args = parser.parse_args()

  print('Metric Recalculation with arguments', args)
  print()

  metric_recalculator = MetricsRecalculator(
    model_name=args.model,
    sampling_config=args.sampling,
    data_split=args.data_split,
    epoch=args.epochs,
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

  metric_recalculator.sample()

  print('Sampling script completed.')

