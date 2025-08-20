import sys
from pathlib import Path
# sys.path.append(str(Path().resolve().parent / 'src'))

import argparse
import json

from config import read_model_config
from trainer import Trainer

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--config', type=str, required=True, help='Name of the model configuration file (without .yaml extension)')
  parser.add_argument('--epochs', type=int, default=None, help='Number of epochs to train the model. If None, use the config value.')
  args = parser.parse_args()

  print('Training with arguments', args)
  print()

  config = read_model_config(args.config)
  print('Configuration')
  print(json.dumps(config, indent=2))
  print()

  trainer = Trainer(config, n_epochs=args.epochs)
  trainer.train()

  print('Training completed successfully.')

