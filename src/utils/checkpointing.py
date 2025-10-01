import torch
from pathlib import Path

CHECKPOINTS_PATH = Path(__file__).parents[1] / 'checkpoints'


def load_last_checkpoint(model_name):
  checkpoint_files = list((CHECKPOINTS_PATH / model_name).glob('*.pt'))
  if not checkpoint_files:
    return None

  return torch.load(max(checkpoint_files, key=lambda f: int(f.stem)))

def load_checkpoint(model_name, epoch):
  checkpoint_path = CHECKPOINTS_PATH / model_name / f'{epoch}.pt'
  if not checkpoint_path.exists():
    raise FileNotFoundError(f'Checkpoint for epoch {epoch} not found at {checkpoint_path}')
  return torch.load(checkpoint_path)

def save_checkpoint(model_name, checkpoint_data):
  (CHECKPOINTS_PATH / model_name).mkdir(parents=True, exist_ok=True)
  checkpoint_path = CHECKPOINTS_PATH / model_name / f'{checkpoint_data['epoch']}.pt'
  torch.save(checkpoint_data, checkpoint_path)

def cleanup_checkpoints(model_name, checkpoint_interval):
  checkpoint_files = list((CHECKPOINTS_PATH / model_name).glob('*.pt'))
  checkpoint_files = sorted(checkpoint_files, key=lambda f: int(f.stem))
  checkpoint_files = checkpoint_files[:-1] # keep latest
  for file in checkpoint_files:
    if (int(file.stem) + 1) % checkpoint_interval > 0:
      file.unlink()
