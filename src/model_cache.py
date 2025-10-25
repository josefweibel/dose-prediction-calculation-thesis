from dataclasses import dataclass
from typing import Dict


@dataclass
class CacheConfig:
  intervals: Dict[str, int] # e.g. {"enc4": 4, "enc3": 2}
  warmup: int = 0  # number of initial steps with no cache (optional)


class DeepCache:
  def __init__(self, cfg: CacheConfig):
    self.cfg = cfg
    self.cache = {}  # layer_name -> (feature, last_step)
    self.current_step = None

  def clear(self):
    self.cache.clear()

  def get(self, layer: str):
    return self.cache.get(layer, (None, None))[0]

  def set(self, layer: str, feat: torch.Tensor):
    self.cache[layer] = (feat.detach(), self.current_step)

  def set_current_step(self, step: int):
    self.current_step = step

  def should_use(self, layer: str) -> bool:
    if layer not in self.cfg.intervals.keys():
      return False

    if self.current_step < self.cfg.warmup:
      return False

    feat, last_step = self.cache.get(layer, (None, None))
    if feat is None:
      return False

    interval = self.cfg.intervals.get(layer, 1)
    return (self.current_step - last_step) < interval

  def get_or_compute(self, layer: str, compute_fn):
    if self.should_use(layer):
      return self.get(layer)
    else:
      feat = compute_fn()
      self.set(layer, feat)
      return feat
