import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from multiprocessing import Manager
from pathlib import Path
import pandas as pd

from config import read_model_config
from data import GDPHMMSlicesDataset, SharedLRUCache
from evaluator import Evaluator
from models import MambaUnet
from utils.mp import detect_device, PrefetchLoader
from utils.checkpointing import load_last_checkpoint, load_checkpoint


N_SAMPLES_FOR_TIME_ESTIMATE = 5

SAMPLES_PATH = Path(__file__).parents[1] / 'samples'


class Sampler:
  def __init__(
    self,
    model_name,
    sampling_config,
    data_split,
    epoch=None,
    rank=None,
    world_size=None,
    local_rank=None,
  ):
    self.model_name = model_name
    self.model_config = read_model_config(model_name)
    self.sampling_config = sampling_config
    self.data_split = data_split
    self.rank = rank
    self.world_size = world_size if world_size is not None else 1
    self.is_distributed = rank is not None and world_size is not None
    self.is_main_process = not self.is_distributed or rank == 0

    self.device = detect_device(local_rank)

    self.manager = Manager()
    dummy_cache = SharedLRUCache(0, self.manager)
    self.dataset = GDPHMMSlicesDataset(
      dataset_name=self.model_config['data']['dataset_name'],
      split=data_split,
      img_size=self.model_config['data']['img_size'],
      cache=dummy_cache,
    )
    if self.is_main_process:
      print('Dataset size', len(self.dataset))

    self.sampler = None
    if self.is_distributed:
      self.sampler = DistributedSampler(
        self.dataset,
        num_replicas=world_size,
        rank=rank,
      )

    self.dataloader = DataLoader(
      self.dataset,
      batch_size=self.sampling_config['batch_size'] // self.world_size,
      shuffle=False,
      sampler=self.sampler,
      num_workers=self.sampling_config['num_workers'],
      pin_memory=True,
      prefetch_factor=self.sampling_config['prefetch_factor'],
      persistent_workers=self.sampling_config['num_workers'] > 0,
    )

    if self.sampling_config['prefetch_on_gpu'] and self.device.type == 'cuda':
      self.dataloader = PrefetchLoader(self.dataloader, self.device)

    self.ema_model = MambaUnet(
      img_size=self.model_config['data']['img_size'],
      in_struc_channels=self.dataset.n_structures
    ).to(self.device)
    self.underlying_model = self.ema_model

    if self.is_distributed:
      self.ema_model = DDP(
        self.ema_model,
        device_ids=[self.rank] if self.device.type == 'cuda' else None,
        find_unused_parameters=False
      )

    if epoch is None:
      checkpoint = load_last_checkpoint(model_name)
    else:
      checkpoint = load_checkpoint(model_name, epoch)

    self.epoch = checkpoint['epoch']
    self.ema_model.load_state_dict(checkpoint['ema_model_state_dict'])

    self.evaluator = Evaluator(
      self.dataloader,
      self.model_config,
      self.sampling_config,
      self.device,
      rank=self.rank,
      world_size=self.world_size,
    )

  def sample(self):
    dir = SAMPLES_PATH / self.model_name / self.sampling_config['name'] / f'{self.data_split}_{self.epoch}'
    self.underlying_model.eval()
    _ = self.evaluator.evaluate(self.underlying_model, dir=dir)

    if self.is_main_process:
      for i in range(N_SAMPLES_FOR_TIME_ESTIMATE):
        pass


