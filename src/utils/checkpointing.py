import torch
from pathlib import Path

CHECKPOINTS_PATH = Path(__file__).parents[2] / 'checkpoints'


def load_last_checkpoint(model_name):
  checkpoint_files = list((CHECKPOINTS_PATH / model_name).glob('*.pt'))
  if not checkpoint_files:
    return None

  return torch.load(max(checkpoint_files, key=lambda f: int(f.stem)), map_location=torch.device('mps'))

def find_checkpoint(model_name, epoch):
  """Find the path to a checkpoint file for a specific epoch."""
  checkpoint_path = CHECKPOINTS_PATH / model_name / f'{epoch}.pt'
  if not checkpoint_path.exists():
    raise FileNotFoundError(f'Checkpoint for epoch {epoch} not found at {checkpoint_path}')
  return checkpoint_path

def load_checkpoint(model_name, epoch):
  return torch.load(find_checkpoint(model_name, epoch), map_location=torch.device('mps'))

def save_checkpoint(model_name, checkpoint_data):
  (CHECKPOINTS_PATH / model_name).mkdir(parents=True, exist_ok=True)
  checkpoint_path = CHECKPOINTS_PATH / model_name / f"{checkpoint_data['epoch']}.pt"
  torch.save(checkpoint_data, checkpoint_path)

def cleanup_checkpoints(model_name, checkpoint_interval):
  checkpoint_files = list((CHECKPOINTS_PATH / model_name).glob('*.pt'))
  checkpoint_files = sorted(checkpoint_files, key=lambda f: int(f.stem))
  checkpoint_files = checkpoint_files[:-1] # keep latest
  for file in checkpoint_files:
    if (int(file.stem) + 1) % checkpoint_interval > 0:
      file.unlink()
