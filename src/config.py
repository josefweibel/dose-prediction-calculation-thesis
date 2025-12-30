import yaml
from pathlib import Path


CONFIG_PATH = Path(__file__).parents[1] / 'configs'

def read_model_config(name):
  """
  Load model config from configs/models/{name}.yaml.

  For distilled models, name can include the distillation path:
    '101-beamdiff/distill-progressive-v1/round-7-8-to-4'

  In this case, we load the base model config ('101-beamdiff')
  and set the name to the full path.
  """
  base_name = get_base_model_name(name)

  with open(CONFIG_PATH / 'models' / f'{base_name}.yaml', 'r') as file:
    config = yaml.safe_load(file)

  config['name'] = name
  return config

def read_sampling_config(name):
  with open(CONFIG_PATH / 'samplers' / f'{name}.yaml', 'r') as file:
    config = yaml.safe_load(file)

  config['name'] = name
  return config

def read_distillation_config(name):
  """Load distillation config from configs/distillation/{name}.yaml."""
  with open(CONFIG_PATH / 'distillation' / f'{name}.yaml', 'r') as file:
    config = yaml.safe_load(file)

  config['name'] = name
  return config

def get_base_model_name(model_name):
  """
  Extract base model name from a potentially distilled model name.

  Examples:
    '101-beamdiff' -> '101-beamdiff'
    '101-beamdiff/distill-progressive-v1/round-7-8-to-4' -> '101-beamdiff'
  """
  return model_name.split('/')[0]

def parse_distilled_model_name(model_name):
  """
  Parse a distilled model name into components.

  Args:
    model_name: Full model name, e.g.:
      '101-beamdiff/distill-progressive-v1/round-7-8-to-4'

  Returns:
    Dictionary with:
      - base_model: Base model name (e.g., '101-beamdiff')
      - distill_config: Distillation config name (e.g., 'progressive-v1') or None
      - round: Round string (e.g., '7-8-to-4') or None
      - round_idx: Round index (e.g., 7) or None
      - teacher_T: Teacher steps (e.g., 8) or None
      - student_T: Student steps (e.g., 4) or None
  """
  parts = model_name.split('/')

  result = {
    'base_model': parts[0],
    'distill_config': None,
    'round': None,
    'round_idx': None,
    'teacher_T': None,
    'student_T': None,
  }

  if len(parts) >= 2 and parts[1].startswith('distill-'):
    result['distill_config'] = parts[1][8:]  # Remove 'distill-' prefix

  if len(parts) >= 3 and parts[2].startswith('round-'):
    round_str = parts[2][6:]  # Remove 'round-' prefix
    result['round'] = round_str

    # Parse round string: 'idx-teacher_T-to-student_T'
    round_parts = round_str.split('-')
    if len(round_parts) >= 4 and round_parts[2] == 'to':
      result['round_idx'] = int(round_parts[0])
      result['teacher_T'] = int(round_parts[1])
      result['student_T'] = int(round_parts[3])

  return result

def build_distilled_model_name(base_model, distill_config, round_str):
  """
  Build a distilled model name from components.

  Args:
    base_model: Base model name (e.g., '101-beamdiff')
    distill_config: Distillation config name (e.g., 'progressive-v1')
    round_str: Round string (e.g., '8-to-4')

  Returns:
    Full model name (e.g., '101-beamdiff/distill-progressive-v1/round-8-to-4')
  """
  return f"{base_model}/distill-{distill_config}/round-{round_str}"

