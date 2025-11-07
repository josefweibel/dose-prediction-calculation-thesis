import torch
from torch.utils.data import DataLoader, DistributedSampler
from multiprocessing import Manager
import numpy as np

from config import read_model_config
from data import GDPHMMSlicesDataset, SharedLRUCache, SharedCache
from evaluator import Evaluator
from models import MambaUnet, remap_state_dict
from utils.mp import detect_device, PrefetchLoader
from utils.checkpointing import load_last_checkpoint, load_checkpoint
from results import SAMPLES_PATH


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
    if self.sampling_config['data_cache_size'] == 'inf':
      shared_cache = SharedCache(self.manager)
    else:
      shared_cache = SharedLRUCache(self.sampling_config['data_cache_size'], self.manager)

    self.dataset = GDPHMMSlicesDataset(
      dataset_name=self.model_config['data']['dataset_name'],
      split=data_split,
      img_size=self.model_config['data']['img_size'],
      cache=shared_cache,
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
      batch_size=self.sampling_config['batch_size_per_gpu'],
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
      in_struc_channels=self.dataset.n_structures,
      num_layers=self.model_config['diffusion'].get('arch', {}).get('num_layers', 4),
    ).to(self.device)

    if epoch is None or epoch == 'latest':
      checkpoint = load_last_checkpoint(model_name)
    else:
      checkpoint = load_checkpoint(model_name, epoch)

    self.epoch = checkpoint['epoch']

    model_state_dict = checkpoint['ema_model_state_dict']
    if "t_proj_enc1.proj.weight" in model_state_dict:
      model_state_dict = remap_state_dict(model_state_dict)
    self.ema_model.load_state_dict(model_state_dict)

    self.evaluator = Evaluator(
      self.dataloader,
      self.model_config,
      self.sampling_config,
      self.device,
      rank=self.rank,
      world_size=self.world_size,
    )

  def sample(self):
    print('Rank', self.rank, 'starts sampling')
    self.set_seed(self.sampling_config['seed'])

    dir = SAMPLES_PATH / self.model_name / self.sampling_config['name'] / f'{self.data_split}_{self.epoch}'

    self.ema_model.eval()
    _ = self.evaluator.evaluate(self.ema_model, dir=dir)

  def finish(self):
    print('finishing sampler instance', flush=True)

    self.dataset.finish()
    self.dataloader = None

    torch.cuda.empty_cache()

  def set_seed(self, seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

