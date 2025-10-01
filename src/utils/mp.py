import torch

def detect_device(local_rank):
  if local_rank is not None:
    return torch.device(f"cuda:{local_rank}")
  elif torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    return torch.device("mps")
  else:
    return torch.device("cpu")


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
