import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import copy
import wandb
from multiprocessing import Manager

from data import GDPHMMSlicesDataset, SharedCache, SharedLRUCache
from models import MambaUnet
from model_cache import DeepCache, CacheConfig
from evaluator import Evaluator
from config import read_sampling_config
from utils.mp import detect_device, PrefetchLoader
from utils.checkpointing import load_last_checkpoint, save_checkpoint, cleanup_checkpoints, CHECKPOINTS_PATH


def extract(v, t, x_shape):
  """
  Extract some coefficients at specified timesteps, then reshape to
  [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
  """
  return v.gather(0, t).view([t.shape[0]] + [1] * (len(x_shape) - 1))


class DiffusionModelTrainerModule(nn.Module):
  def __init__(self, denoising_model, beta_1, beta_T, T):
    super().__init__()

    self.model = denoising_model
    self.T = T

    betas = torch.linspace(beta_1, beta_T, T).float()
    alphas = 1. - betas
    alphas_bar = torch.cumprod(alphas, dim=0)

    self.register_buffer('sqrt_alphas_bar', torch.sqrt(alphas_bar))
    self.register_buffer('sqrt_one_minus_alphas_bar', torch.sqrt(1. - alphas_bar))

    self.cache = DeepCache(CacheConfig(intervals={}, warmup=0, layers=[], sampling_steps=T))

  def forward(self, x_0, condition):
    t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)

    noise = torch.randn_like(x_0)
    x_t = (
      extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
      extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise
    )

    noise_pred = self.model(x_t, condition, t, self.cache)

    return F.mse_loss(noise_pred, noise, reduction='none').mean()

class Trainer:
  def __init__(self, model_config, n_epochs=None, rank=None, world_size=None, local_rank=None):
    self.name = model_config["name"]
    self.config = model_config
    self.rank = rank
    self.local_rank = local_rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    self.device = detect_device(local_rank)

    self.manager = Manager()

    if self.config['data']['train_cache_size'] == 'inf':
      shared_train_cache = SharedCache(self.manager)
    else:
      shared_train_cache = SharedLRUCache(self.config['data']['train_cache_size'], self.manager)

    train_dataset = GDPHMMSlicesDataset(
      dataset_name=self.config['data']['dataset_name'],
      split='train',
      img_size=self.config['data']['img_size'],
      limit=self.config['data']['train_limit'] if 'train_limit' in self.config['data'] else None,
      skip=self.config['data']['train_skip'] if 'train_skip' in self.config['data'] else 0,
      cache=shared_train_cache,
    )
    if self.is_main_process:
      print('Training dataset size', len(train_dataset))

    if self.config['data'].get('val_on_train', False):
      val_dataset = train_dataset
    else:
      if self.config['data']['valid_cache_size'] == 'inf':
        shared_valid_cache = SharedCache(self.manager)
      else:
        shared_valid_cache = SharedLRUCache(self.config['data']['valid_cache_size'], self.manager)

      val_dataset = GDPHMMSlicesDataset(
        dataset_name=self.config['data']['dataset_name'],
        split='valid',
        img_size=self.config['data']['img_size'],
        limit=self.config['data']['valid_limit'] if 'valid_limit' in self.config['data'] else None,
        skip=self.config['data']['valid_skip'] if 'valid_skip' in self.config['data'] else 0,
        cache=shared_valid_cache,
      )
    if self.is_main_process:
      print('Validation dataset size', len(val_dataset))

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
      in_struc_channels=train_dataset.n_structures,
      num_layers=self.config.get('model', {}).get('num_layers', 4),
    ).to(self.device)
    self.ema_model = copy.deepcopy(self.net_model).to(self.device)

    self.model = DiffusionModelTrainerModule(
      denoising_model=self.net_model,
      beta_1=self.config['beta_1'],
      beta_T=self.config['beta_T'],
      T=self.config['T']
    ).to(self.device)

    if self.is_distributed:
      self.model = DDP(
        self.model,
        device_ids=[self.local_rank],
        output_device=self.local_rank,
        find_unused_parameters=False,
      )

    self.__setup_optimizer()
    self.use_mixed_precision = self.device.type == 'cuda' and self.config['mixed_precision']
    self.scaler = torch.amp.GradScaler('cuda') if self.use_mixed_precision else None

    self.epoch = -1

    checkpoint = load_last_checkpoint(self.name)
    if checkpoint:
      self.epoch = checkpoint['epoch']
      if self.is_main_process:
        print('Loading checkpoint and continue with epoch', checkpoint['epoch'] + 1)

      self.net_model.load_state_dict(checkpoint['net_model_state_dict'])
      self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
      self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
      self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

      if self.scaler is not None and 'scaler_state_dict' in checkpoint:
        self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
    elif self.is_main_process:
      print('Start fresh run')

    if n_epochs is None:
      self.to_epoch = self.config['epochs']
    else:
      self.to_epoch = self.epoch + n_epochs + 1

    if self.is_main_process:
      wandb.login()
      self.wandb_run = self.__setup_wandb_run(checkpoint)
    else:
      self.wandb_run = None

    self.evaluator = Evaluator(
      self.val_dataloader,
      model_config,
      read_sampling_config(self.config['validation_sampler']),
      self.device,
      rank=self.rank,
      world_size=self.world_size,
    )

  def train(self):
    pbar_epochs = tqdm(range(self.epoch + 1, self.to_epoch), desc='Training', disable=not self.is_main_process)
    for self.epoch in pbar_epochs:
      if self.is_distributed and self.train_sampler is not None:
        self.train_sampler.set_epoch(self.epoch)

      self.model.train()

      pbar_batches = tqdm(self.train_dataloader, desc=f'Training epoch {self.epoch}', leave=False, disable=not self.is_main_process)
      for structure, target, _, _ in pbar_batches:
        self.optim.zero_grad()
        structure = structure.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)

        with torch.amp.autocast('cuda', dtype=torch.float16) if self.use_mixed_precision else contextlib.nullcontext():
          loss = self.model(target, structure)

        current_lr = self.optim.param_groups[0]['lr']

        if self.scaler is not None:
          self.scaler.scale(loss).backward()
          self.scaler.unscale_(self.optim)
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
          self.scaler.step(self.optim)
          self.scaler.update()
        else:
          loss.backward()
          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
          self.optim.step()
          self.lr_scheduler.step()

        self.__update_ema_weigths(self.net_model, self.ema_model, self.config['ema_decay'])

        if self.is_main_process:
          pbar_epochs.set_postfix(loss=loss.item())
          if self.wandb_run:
            self.wandb_run.log({'train/loss': loss.item(), 'train/lr': current_lr, 'epoch': self.epoch})

      if self.is_distributed:
        dist.barrier()

      if self.is_main_process:
        self.__save_checkpoint()
        cleanup_checkpoints(self.name, self.config['checkpoint_interval'])

      if self.is_distributed:
        dist.barrier()

      if self.epoch == 0 or (self.epoch + 1) % self.config['eval_interval'] == 0:
        for t in self.ema_model.state_dict().values():
          dist.broadcast(t, src=0)

        self.ema_model.eval()
        _, metrics = self.evaluator.evaluate(self.ema_model, dir=CHECKPOINTS_PATH / self.name / f'epoch-{self.epoch}')

        if self.is_main_process and self.wandb_run:
          metrics = {f'eval/{key}': value for key, value in metrics.to_dict().items()}
          self.wandb_run.log({**metrics, 'epoch': self.epoch})

  def finish(self):
    print('finishing trainer instance', flush=True)
    if self.is_main_process and self.wandb_run:
      self.wandb_run.finish()

    if self.train_dataloader is not None:
      self.train_dataloader.dataset.finish()

    if self.val_dataloader is not None:
      self.val_dataloader.dataset.finish()

    self.train_dataloader = None
    self.val_dataloader = None

    torch.cuda.empty_cache()

  def __save_checkpoint(self):
    checkpoint_data = {
      'epoch': self.epoch,
      'net_model_state_dict': self.net_model.state_dict(),
      'ema_model_state_dict': self.ema_model.state_dict(),
      'optimizer_state_dict': self.optim.state_dict(),
      'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
      'wandb_run_id': self.wandb_run.id
    }

    if self.scaler is not None:
      checkpoint_data['scaler_state_dict'] = self.scaler.state_dict()

    save_checkpoint(self.name, checkpoint_data)

  def __update_ema_weigths(self, source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
      target_dict[key].data.copy_(
        target_dict[key].data * decay + source_dict[key].data * (1 - decay)
      )

  def __setup_wandb_run(self, checkpoint):
    if checkpoint:
      return wandb.init(
        project='dose-prediction-thesis',
        entity='joseph-weibel',
        name=self.name,
        id=checkpoint['wandb_run_id'],
        config=self.config,
        resume='must',
      )
    else:
      return wandb.init(
        project='dose-prediction-thesis',
        entity='joseph-weibel',
        name=self.name,
        id=None,
        config=self.config,
      )

  def __setup_optimizer(self):
    if 'lr_scheduler' in self.config:
      start_decay = self.config['lr_scheduler']['start_decay']
      end_decay = self.config['lr_scheduler']['end_decay']
      end_lr = self.config['lr_scheduler']['end_lr']
      start_lr = self.config['lr_scheduler']['start_lr']

      def linear_lr_schedule(step):
        if step < start_decay:
          return 1.0
        else:
          progress = (step - start_decay) / (end_decay - start_decay)
          scale = (end_lr / start_lr) + (1 - progress) * (1 - end_lr / start_lr)
          return max(scale, end_lr / start_lr)

      self.optim = torch.optim.AdamW(self.model.parameters(), lr=start_lr)
      self.lr_scheduler = LambdaLR(self.optim, lr_lambda=linear_lr_schedule)
    else:
      self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
      self.lr_scheduler = LambdaLR(self.optim, lr_lambda=lambda step: 1.0)
