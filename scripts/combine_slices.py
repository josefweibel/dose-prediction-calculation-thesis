import argparse
import signal
import sys
import atexit

from slice_combinator import SliceCombinator


if __name__ == '__main__':
  def parse_epoch(value):
    if value == "latest":
      return value
    try:
      return int(value)
    except ValueError:
      raise argparse.ArgumentTypeError("epoch must be an integer or 'latest'")

  parser = argparse.ArgumentParser(description='Combine individual slice files into volume files')
  parser.add_argument('--model', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--sampling', type=str, required=True, help='Name of the sampling configuration file (without .yaml extension)')
  parser.add_argument('--data-split', type=str, choices=['train', 'valid', 'test'], default='test', help='Data split to use for sampling.')
  parser.add_argument('--epoch', type=parse_epoch, default=None, help='Epoch number to use or "latest".')
  parser.add_argument('--delete-slices', action='store_true', help='Delete individual slice files after combining into volumes.')
  args = parser.parse_args()

  print('Combining slices with arguments', args)
  print()

  combinator = SliceCombinator(
    model_name=args.model,
    sampling_name=args.sampling,
    data_split=args.data_split,
    epoch=args.epoch,
    delete_slices=args.delete_slices,
  )

  def cleanup():
    print('cleanup of slice combinator tried')
    combinator.finish()

  atexit.register(cleanup)

  def handle_signal(signum, frame):
    print(f"Received signal {signum}")
    cleanup()
    sys.exit(0)

  signal.signal(signal.SIGTERM, handle_signal)
  signal.signal(signal.SIGINT, handle_signal)

  combinator.combine()

  print('Slice combining script completed.')
