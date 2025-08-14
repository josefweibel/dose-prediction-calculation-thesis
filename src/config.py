import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).parents[2] / 'configs'

def read_model_config(name):
  with open(CONFIG_PATH / 'models' / f'{name}.yaml', 'r') as file:
    config = yaml.safe_load(file)

  config['name'] = name
  return config
