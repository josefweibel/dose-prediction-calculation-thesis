from pathlib import Path
import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from tqdm import tqdm
import copy
import wandb
from multiprocessing import Manager

from data import GDPHMMSlicesDataset, SharedLRUCache
from models import MambaUnet
from evaluator import Evaluator
from config import read_sampling_config

CHECKPOINTS_PATH = Path(__file__).parents[1] / 'checkpoints'


def extract(v, t, x_shape):
  """
  Extract some coefficients at specified timesteps, then reshape to
  [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
  """
  out = v.to(t.device).gather(0, t).float()
  return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DiffusionModelTrainerModule(nn.Module):
  def __init__(self, denoising_model, beta_1, beta_T, T):
    super().__init__()

    self.model = denoising_model
    self.T = T

    self.register_buffer('betas', torch.linspace(beta_1, beta_T, T).float())
    alphas = 1. - self.betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
    self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

  def forward(self, x_0, condition):
    t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
    noise = torch.randn_like(x_0)
    x_t = (
      extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
      extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
    )

    noise_pred = self.model(x_t, condition, t)
    return F.mse_loss(noise_pred, noise, reduction='none').mean()

class Trainer:
  def __init__(self, model_config, n_epochs=None, rank=None, world_size=None):
    self.name = model_config["name"]
    self.config = model_config
    self.rank = rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    self.device = self.__detect_device()

    self.manager = Manager()
    shared_train_cache = SharedLRUCache(self.config['data']['train_cache_size'], manager=self.manager)
    shared_valid_cache = SharedLRUCache(self.config['data']['valid_cache_size'], manager=self.manager)

    train_dataset = GDPHMMSlicesDataset(
      dataset_name=self.config['data']['dataset_name'],
      split='train',
      img_size=self.config['data']['img_size'],
      limit=self.config['data']['train_limit'] if 'train_limit' in self.config['data'] else None,
      cache=shared_train_cache,
    )
    if self.is_main_process:
      print('Training dataset size', len(train_dataset))

    if 'val_on_train' in self.config['data'] and self.config['data']['val_on_train']:
      val_dataset = train_dataset
    else:
      val_dataset = GDPHMMSlicesDataset(
        dataset_name=self.config['data']['dataset_name'],
        split='valid',
        img_size=self.config['data']['img_size'],
        limit=self.config['data']['valid_limit'] if 'valid_limit' in self.config['data'] else None,
        cache=shared_valid_cache,
      )
    if self.is_main_process:
      print('Validation dataset size', len(val_dataset))

    # Setup distributed samplers
    self.train_sampler = None
    self.val_sampler = None
    if self.is_distributed:
      self.train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
      )
      self.val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
      )

    self.train_dataloader = DataLoader(
      train_dataset,
      batch_size=self.config['batch_size'] // self.world_size,
      shuffle=self.train_sampler is None,  # Only shuffle if not using distributed sampler
      sampler=self.train_sampler,
      num_workers=self.config['num_workers'],
      pin_memory=True,
      prefetch_factor=self.config['prefetch_factor'],
      persistent_workers=self.config['num_workers'] > 0,
    )
    self.val_dataloader = DataLoader(
      val_dataset,
      batch_size=self.config['batch_size'],
      shuffle=False,
      sampler=self.val_sampler,
      num_workers=self.config['num_workers'],
      pin_memory=True,
      prefetch_factor=self.config['prefetch_factor'],
      persistent_workers=self.config['num_workers'] > 0,
    )

    if self.config['prefetch_on_gpu'] and self.device.type == 'cuda':
      self.train_dataloader = PrefetchLoader(self.train_dataloader, self.device)
      self.val_dataloader = PrefetchLoader(self.val_dataloader, self.device)

    self.net_model = MambaUnet(
      img_size=self.config['data']['img_size'],
      in_struc_channels=train_dataset.n_structures
    ).to(self.device)

    if self.is_distributed:
      self.net_model = DDP(
        self.net_model,
        device_ids=[self.rank] if self.device.type == 'cuda' else None,
        find_unused_parameters=False
      )

    self.ema_model = copy.deepcopy(self.underlying_model).to(self.device)

    self.model = DiffusionModelTrainerModule(
      denoising_model=self.underlying_model,
      beta_1=self.config['beta_1'],
      beta_T=self.config['beta_T'],
      T=self.config['T']
    ).to(self.device)

    self.optim = torch.optim.AdamW(self.underlying_model.parameters(), lr=self.config['lr'])
    self.use_mixed_precision = self.device.type == 'cuda' and self.config['mixed_precision']
    self.scaler = torch.amp.GradScaler('cuda') if self.use_mixed_precision else None

    self.epoch = -1

    checkpoint = self.__load_last_checkpoint()
    if checkpoint:
      self.epoch = checkpoint['epoch']
      if self.is_main_process:
        print('Loading checkpoint and continue with epoch', checkpoint['epoch'] + 1)

      self.underlying_model.load_state_dict(checkpoint['net_model_state_dict'])
      self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
      self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

      if self.scaler is not None and 'scaler_state_dict' in checkpoint:
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    else:
      if self.is_main_process:
        print('Start fresh run')
        (CHECKPOINTS_PATH / self.name).mkdir(parents=True, exist_ok=True)

    if n_epochs is None:
      self.to_epoch = self.config['epochs']
    else:
      self.to_epoch = self.epoch + n_epochs + 1

    if self.is_main_process:
      wandb.login()
      self.wandb_run = self.__setup_wandb_run(checkpoint)
    else:
      self.wandb_run = None

    sampling_config = read_sampling_config(self.config['validation_sampler'])
    self.evaluator = Evaluator(self.val_dataloader, model_config, sampling_config, self.device)

  @property
  def underlying_model(self):
    """Get the underlying model without DDP wrapper."""
    return self.net_model.module if self.is_distributed else self.net_model

  def train(self):
    pbar_epochs = tqdm(range(self.epoch + 1, self.to_epoch), desc='Training', disable=not self.is_main_process)
    for self.epoch in pbar_epochs:
      if self.is_distributed and self.train_sampler is not None:
        self.train_sampler.set_epoch(self.epoch)

      self.net_model.train()

      pbar_batches = tqdm(self.train_dataloader, desc=f'Training epoch {self.epoch}', leave=False, disable=not self.is_main_process)
      for structure, target, _ in pbar_batches:
        self.optim.zero_grad()
        structure = structure.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.autocast('cuda', dtype=torch.float16) if self.use_mixed_precision else contextlib.nullcontext():
          loss = self.model(target, structure)

        # Scale loss and backward pass
        if self.scaler is not None:
          self.scaler.scale(loss).backward()
          self.scaler.unscale_(self.optim)
          torch.nn.utils.clip_grad_norm_(self.underlying_model.parameters(), self.config['grad_clip'])
          self.scaler.step(self.optim)
          self.scaler.update()
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.underlying_model.parameters(), self.config['grad_clip'])
          self.optim.step()

        self.__update_ema_weigths(self.underlying_model, self.ema_model, self.config['ema_decay'])

        if self.is_main_process:
          pbar_epochs.set_postfix(loss=loss.item())
          if self.wandb_run:
            self.wandb_run.log({'train/loss': loss.item(), 'epoch': self.epoch})

      if self.is_distributed:
        dist.barrier()

      if self.is_main_process:
        self.__save_checkpoint()
        self.__cleanup_checkpoints()

      if self.is_distributed:
        dist.barrier()

      if self.epoch == 0 or (self.epoch + 1) % self.config['eval_interval'] == 0:
        self.ema_model.eval()  # Ensure model is in eval mode
        if self.is_main_process:
          metrics = self.evaluator.evaluate(self.ema_model, dir=CHECKPOINTS_PATH / self.name / f'epoch_{self.epoch}')
          metrics = {f'eval/{key}': value for key, value in metrics.items()}
          if self.wandb_run:
            self.wandb_run.log({**metrics, 'epoch': self.epoch})

    if self.is_main_process and self.wandb_run:
      self.wandb_run.finish()

  def __load_last_checkpoint(self):
    checkpoint_files = list((CHECKPOINTS_PATH / self.name).glob('*.pt'))
    if not checkpoint_files:
      return None

    return torch.load(max(checkpoint_files, key=lambda f: int(f.stem)))

  def __save_checkpoint(self):
    checkpoint_path = CHECKPOINTS_PATH / self.name / f'{self.epoch}.pt'

    checkpoint_data = {
      'epoch': self.epoch,
      'net_model_state_dict': self.underlying_model.state_dict(),
      'ema_model_state_dict': self.ema_model.state_dict(),
      'optimizer_state_dict': self.optim.state_dict(),
      'wandb_run_id': self.wandb_run.id
    }

    if self.scaler is not None:
      checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

    torch.save(checkpoint_data, checkpoint_path)

  def __cleanup_checkpoints(self):
    checkpoint_files = list((CHECKPOINTS_PATH / self.name).glob('*.pt'))
    checkpoint_files = sorted(checkpoint_files, key=lambda f: int(f.stem))
    checkpoint_files = checkpoint_files[:-1] # keep latest
    for file in checkpoint_files:
      if (int(file.stem) + 1) % self.config['checkpoint_interval'] > 0:
        file.unlink()

  def __update_ema_weigths(self, source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
      target_dict[key].data.copy_(
        target_dict[key].data * decay + source_dict[key].data * (1 - decay)
      )

  def __detect_device(self):
    if self.is_distributed:
      # In distributed mode, use the specified rank as device
      return torch.device(f"cuda:{self.rank}")
    elif torch.cuda.is_available():
      return torch.device("cuda")
    elif torch.backends.mps.is_available():
      return torch.device("mps")
    else:
      return torch.device("cpu")

  def __setup_wandb_run(self, checkpoint):
    if checkpoint:
      return wandb.init(
        project='dose-prediction-thesis',
        name=self.name,
        id=checkpoint['wandb_run_id'],
        config=self.config,
        resume='must',
      )
    else:
      return wandb.init(
        project='dose-prediction-thesis',
        name=self.name,
        id=None,
        config=self.config,
      )

class PrefetchLoader:
  def __init__(self, loader, device):
    self.loader = loader
    self.device = device
    self.stream = torch.cuda.Stream()

  def __getattribute__(self, name):
    if name in ['loader', 'device', 'stream', 'iter', 'next_batch', 'preload', '__iter__', '__next__']:
      return object.__getattribute__(self, name)
    return getattr(self.loader, name)

  def __iter__(self):
    self.iter = iter(self.loader)
    self.preload()
    return self

  def __next__(self):
    torch.cuda.current_stream().wait_stream(self.stream)
    batch = self.next_batch
    if batch is None:
      raise StopIteration
    self.preload()
    return batch

  def __len__(self):
    return len(self.loader)

  def preload(self):
    try:
      batch = next(self.iter)
    except StopIteration:
      self.next_batch = None
      return
    with torch.cuda.stream(self.stream):
      self.next_batch = tuple(
        x.to(self.device, non_blocking=True) if torch.is_tensor(x) else x
        for x in batch
      )
