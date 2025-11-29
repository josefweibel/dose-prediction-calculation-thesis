import contextlib
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR, StepLR
from tqdm import tqdm
import copy
import wandb

from data import create_gdphmm_dataset
from model_cache import DeepCache, CacheConfig
from evaluator import Evaluator
from config import read_sampling_config
from utils.mp import detect_device, PrefetchLoader
from utils.checkpointing import load_last_checkpoint, save_checkpoint, cleanup_checkpoints, CHECKPOINTS_PATH
from c3d_model import C3DLoss
from models import build_model

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

    return F.mse_loss(noise_pred, noise, reduction='mean')


class MedNeXtUnetTrainerModule(nn.Module):
  def __init__(self, model):
    super().__init__()

    self.model = model

  def forward(self, x, condition):
    output = self.model(condition)
    output = F.relu(output)
    return F.l1_loss(output, x, reduction='mean')


class C3DTrainerModule(nn.Module):
  def __init__(self, model, alpha=0.5):
    super().__init__()

    self.model = model
    self.loss_fn = C3DLoss(alpha=alpha)

  def forward(self, x, condition):
    # C3D model returns (output_a, output_b) during training
    output_a, output_b = self.model(condition)
    return self.loss_fn(output_a, output_b, x)


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

    # Training dataset uses only conditioning groups
    conditioning_groups = self.config.get('conditioning', {}).get('types', ['ct', 'ptv', 'oar'])
    train_dataset = create_gdphmm_dataset(self.config, 'train', structure_groups=conditioning_groups)

    if self.config['data'].get('val_on_train', False):
      val_dataset = train_dataset
    else:
      # Validation dataset loads all structures for metrics computation
      val_dataset = create_gdphmm_dataset(self.config, 'valid', structure_groups=None)

    if self.is_main_process:
      print('Training dataset size', len(train_dataset))
      print('Validation dataset size', len(val_dataset))
      print(f'\n[DEBUG] Dataset configuration:')
      print(f'  Training dataset groups: {train_dataset.selector.groups}')
      print(f'  Training dataset total_channels: {train_dataset.selector.total_channels}')
      print(f'  Validation dataset groups: {val_dataset.selector.groups}')
      print(f'  Validation dataset total_channels: {val_dataset.selector.total_channels}')
      print(f'  Conditioning groups: {conditioning_groups}')

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

    self.net_model = build_model(self.config, train_dataset.selector.total_channels).to(self.device)

    if 'diffusion' in self.config:
      self.model = DiffusionModelTrainerModule(
        denoising_model=self.net_model,
        beta_1=self.config['diffusion']['beta_1'],
        beta_T=self.config['diffusion']['beta_T'],
        T=self.config['diffusion']['T']
      ).to(self.device)
    elif 'c3d' in self.config:
      alpha = self.config['c3d'].get('alpha', 0.5)
      self.model = C3DTrainerModule(self.net_model, alpha=alpha).to(self.device)
    else:
      self.model = MedNeXtUnetTrainerModule(self.net_model).to(self.device)

    self.ema_model = copy.deepcopy(self.net_model).to(self.device)

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

      # Load state dicts on all ranks
      self.net_model.load_state_dict(checkpoint['net_model_state_dict'])
      self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])
      self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
      self.optim.load_state_dict(checkpoint['optimizer_state_dict'])

      # CRITICAL: Synchronize ALL state across ranks in distributed training
      # This ensures all ranks have identical weights/buffers/optimizer state after loading
      if self.is_distributed:
        # Get the unwrapped model (in case of DDP)
        model_to_sync = self.model.module if hasattr(self.model, 'module') else self.model

        # Broadcast all parameters and buffers from rank 0 to all other ranks
        # This includes: model weights, batch norm stats, diffusion alphas, etc.
        for param in model_to_sync.parameters():
          dist.broadcast(param.data, src=0)
        for buffer in model_to_sync.buffers():
          dist.broadcast(buffer.data, src=0)

        # Also sync EMA model
        for param in self.ema_model.parameters():
          dist.broadcast(param.data, src=0)
        for buffer in self.ema_model.buffers():
          dist.broadcast(buffer.data, src=0)

        # CRITICAL: Also sync optimizer state (momentum, velocity buffers, etc.)
        # This is crucial - optimizer state must be identical across ranks
        for state in self.optim.state.values():
          if isinstance(state, dict):
            for key, value in state.items():
              if torch.is_tensor(value):
                dist.broadcast(value, src=0)

        # CRITICAL: Also sync LR scheduler state
        # The scheduler has internal state like last_epoch, step counts, etc.
        # Use broadcast_object_list to sync the entire state dict from rank 0
        scheduler_state = [self.lr_scheduler.state_dict() if self.is_main_process else None]
        dist.broadcast_object_list(scheduler_state, src=0)

        # Reload the synchronized scheduler state on non-main ranks
        if not self.is_main_process:
          self.lr_scheduler.load_state_dict(scheduler_state[0])

        # Synchronize all ranks before continuing
        dist.barrier()

        if self.is_main_process:
          print('Model weights, buffers, optimizer state, and LR scheduler synchronized across all ranks')

      # Debug: Check for NaN/Inf in loaded model weights on ALL ranks (not just main)
      # This helps catch if synchronization failed
      if self.is_distributed:
        rank_str = f'[rank{self.rank}]'
        has_nan = any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.net_model.parameters())
        if has_nan:
          print(f'{rank_str} ERROR: NaN/Inf detected in model after loading checkpoint!')
        else:
          print(f'{rank_str} Model weights check: OK (no NaN/Inf)')

        # Also check optimizer state for NaN
        optimizer_has_nan = False
        for state in self.optim.state.values():
          if isinstance(state, dict):
            for key, value in state.items():
              if torch.is_tensor(value) and (torch.isnan(value).any() or torch.isinf(value).any()):
                optimizer_has_nan = True
                print(f'{rank_str} ERROR: NaN/Inf detected in optimizer state key: {key}')
        if not optimizer_has_nan:
          print(f'{rank_str} Optimizer state check: OK (no NaN/Inf)')

        # Verify LR is the same across ranks
        current_lr = self.optim.param_groups[0]['lr']
        print(f'{rank_str} Learning rate after sync: {current_lr}')

      # Debug: Check for NaN/Inf in loaded model weights
      if self.is_main_process:
        has_nan = False
        has_inf = False
        for name, param in self.net_model.named_parameters():
          if torch.isnan(param).any():
            print(f'WARNING: NaN detected in model parameter: {name}')
            has_nan = True
          if torch.isinf(param).any():
            print(f'WARNING: Inf detected in model parameter: {name}')
            has_inf = True
        if not has_nan and not has_inf:
          print('Model weights check: OK (no NaN/Inf)')

        # Check optimizer state
        for group_idx, group in enumerate(self.optim.param_groups):
          print(f'Optimizer group {group_idx} LR: {group["lr"]}')

        # Check optimizer state for NaN/Inf
        optimizer_has_nan = False
        for state_value in self.optim.state.values():
          if isinstance(state_value, dict):
            for key, val in state_value.items():
              if torch.is_tensor(val):
                if torch.isnan(val).any():
                  print(f'WARNING: NaN detected in optimizer state: {key}')
                  optimizer_has_nan = True
                if torch.isinf(val).any():
                  print(f'WARNING: Inf detected in optimizer state: {key}')
                  optimizer_has_nan = True
        if not optimizer_has_nan:
          print('Optimizer state check: OK (no NaN/Inf)')

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
      for batch_idx, (structure, target, _) in enumerate(pbar_batches):
        self.optim.zero_grad()
        structure = structure.to(self.device, non_blocking=True)
        target = target.to(self.device, non_blocking=True)
        condition = structure # Training dataset already loads only conditioning groups, so structure IS the condition

        # Debug: Print progress on ALL ranks for first few batches to see if there's a distributed training issue
        if batch_idx < 5:  # Always check first 5 batches
          rank_str = f'[rank{self.rank}]' if self.is_distributed else ''
          print(f'{rank_str} Processing batch {batch_idx}, target shape: {target.shape}, condition shape: {condition.shape}')

        # Debug: Print shapes on first batch of first epoch
        if self.epoch == 0 and batch_idx == 0 and self.is_main_process:
          print(f"\n[DEBUG] Training epoch 0, batch 0:")
          print(f"  structure.shape: {structure.shape}")
          print(f"  condition.shape: {condition.shape}")
          print(f"  Expected model input channels: {self.net_model.in_channels if hasattr(self.net_model, 'in_channels') else 'unknown'}")

          # Check condition data statistics
          print(f"\n[DEBUG] Condition statistics (first sample in batch):")
          print(f"  CT channel - min: {condition[0, 0].min().item():.4f}, max: {condition[0, 0].max().item():.4f}, mean: {condition[0, 0].mean().item():.4f}")
          print(f"  PTV_SDM channel - min: {condition[0, 1].min().item():.4f}, max: {condition[0, 1].max().item():.4f}, mean: {condition[0, 1].mean().item():.4f}")
          print(f"  OAR_SDM first channel - min: {condition[0, 2].min().item():.4f}, max: {condition[0, 2].max().item():.4f}, mean: {condition[0, 2].mean().item():.4f}")

        with torch.amp.autocast('cuda', dtype=torch.float16) if self.use_mixed_precision else contextlib.nullcontext():
          loss = self.model(target, condition)

        current_lr = self.optim.param_groups[0]['lr']

        # Debug: Check for NaN loss
        if torch.isnan(loss) or torch.isinf(loss):
          # Print on ALL ranks during debugging, not just main process
          rank_str = f'[rank{self.rank}]' if self.is_distributed else ''
          print(f'\n{rank_str} NaN/Inf loss detected at epoch {self.epoch}, batch {batch_idx}')
          print(f'{rank_str} Loss value: {loss.item()}')
          print(f'{rank_str} Learning rate: {current_lr}')
          print(f'{rank_str} Target stats - min: {target.min().item()}, max: {target.max().item()}, mean: {target.mean().item()}')
          print(f'{rank_str} Condition stats - min: {condition.min().item()}, max: {condition.max().item()}, mean: {condition.mean().item()}')
          # Check model parameters for NaN
          print(f'{rank_str} Checking parameters for NaN/Inf...')
          nan_params = []
          for name, param in self.net_model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
              nan_params.append(name)
          if nan_params:
            print(f'{rank_str} NaN/Inf found in {len(nan_params)} parameters (first 10): {nan_params[:10]}')
          else:
            print(f'{rank_str} No NaN/Inf in parameters - NaN occurred during forward pass!')
          raise ValueError(f'NaN/Inf loss detected at epoch {self.epoch}, batch {batch_idx}')

        if self.scaler is not None:
          self.scaler.scale(loss).backward()
          self.scaler.unscale_(self.optim)

          # Check gradients for NaN/Inf BEFORE clipping and stepping
          if batch_idx < 5:  # Always check first 5 batches
            rank_str = f'[rank{self.rank}]' if self.is_distributed else ''
            for name, param in self.net_model.named_parameters():
              if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f'\n{rank_str} ERROR: NaN/Inf gradient detected at epoch {self.epoch}, batch {batch_idx}')
                print(f'{rank_str} Parameter: {name}')
                print(f'{rank_str} Grad stats - min: {param.grad.min().item()}, max: {param.grad.max().item()}')
                print(f'{rank_str} NaN count: {torch.isnan(param.grad).sum().item()}, Inf count: {torch.isinf(param.grad).sum().item()}')
                raise ValueError(f'NaN/Inf in gradients at epoch {self.epoch}, batch {batch_idx}, param {name}')

          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
          self.scaler.step(self.optim)
          self.scaler.update()
        else:
          loss.backward()

          # Check gradients for NaN/Inf BEFORE clipping and stepping
          if batch_idx < 5:  # Always check first 5 batches
            rank_str = f'[rank{self.rank}]' if self.is_distributed else ''
            for name, param in self.net_model.named_parameters():
              if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                print(f'\n{rank_str} ERROR: NaN/Inf gradient detected at epoch {self.epoch}, batch {batch_idx}')
                print(f'{rank_str} Parameter: {name}')
                print(f'{rank_str} Grad stats - min: {param.grad.min().item()}, max: {param.grad.max().item()}')
                print(f'{rank_str} NaN count: {torch.isnan(param.grad).sum().item()}, Inf count: {torch.isinf(param.grad).sum().item()}')
                print(f'{rank_str} Input data stats:')
                print(f'{rank_str}   Target - min: {target.min().item()}, max: {target.max().item()}, mean: {target.mean().item()}')
                print(f'{rank_str}   Condition - min: {condition.min().item()}, max: {condition.max().item()}, mean: {condition.mean().item()}')
                raise ValueError(f'NaN/Inf in gradients at epoch {self.epoch}, batch {batch_idx}, param {name}')

          torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
          self.optim.step()

        self.lr_scheduler.step()

        # Debug: Check if weights became NaN after optimizer step
        if batch_idx < 5:  # Always check first 5 batches
          rank_str = f'[rank{self.rank}]' if self.is_distributed else ''
          has_nan_after = any(torch.isnan(p).any() or torch.isinf(p).any() for p in self.net_model.parameters())
          if has_nan_after:
            print(f'{rank_str} WARNING: NaN detected in weights AFTER batch {batch_idx} optimizer step!')
            # Find which parameters became NaN
            for name, param in self.net_model.named_parameters():
              if torch.isnan(param).any() or torch.isinf(param).any():
                print(f'{rank_str}   - {name} has NaN/Inf')
          else:
            print(f'{rank_str} Batch {batch_idx} completed, weights OK')

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
    if 'lr_scheduler' in self.config and self.config['lr_scheduler'].get('type', 'linear_decay') == 'linear_decay':
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
    elif 'lr_scheduler' in self.config and self.config['lr_scheduler'].get('type', '') == 'cosine_annealing':
      self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr_scheduler']['lr'])
      self.lr_scheduler = CosineAnnealingLR(
        self.optim,
        T_max=len(self.train_dataloader) * self.config['epochs'],
      )
    elif 'lr_scheduler' in self.config and self.config['lr_scheduler'].get('type', '') == 'step_decay':
      initial_lr = self.config['lr_scheduler']['initial_lr']
      step_size = self.config['lr_scheduler']['step_size']
      gamma = self.config['lr_scheduler']['gamma']

      self.optim = torch.optim.AdamW(self.model.parameters(), lr=initial_lr)
      self.lr_scheduler = StepLR(
        self.optim,
        step_size=step_size * len(self.train_dataloader),
        gamma=gamma
      )
    else:
      self.optim = torch.optim.AdamW(self.model.parameters(), lr=self.config['lr'])
      self.lr_scheduler = LambdaLR(self.optim, lr_lambda=lambda step: 1.0)
