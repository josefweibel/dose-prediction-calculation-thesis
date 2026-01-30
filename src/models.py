from mamba_model import MambaUnet
from mednext import build_mednext_model
from c3d_model import C3D, C3D_2D
from beamdiff_model import build_beamdiff
from dosediff_model import build_dosediff
from deeplabv3plus_model import build_deeplabv3plus
from diffdp_model import build_diffdp
from spdiffdose_model import build_spdiffdose
from hdunet_model import build_hdunet
from analytical_dose import AnalyticalDoseModel


def build_model(model_config, structure_selector, out_channels=1):
  """
  Build a model from configuration.

  Args:
    model_config: Model configuration dict
    structure_selector: StructureSelector instance
    out_channels: Number of output channels (default: 1 for dose prediction,
                  use 2 for warm-start model predicting μ and log_σ)

  Returns:
    Model instance
  """
  n_input_channels = structure_selector.total_channels

  if 'c3d' in model_config:
    if model_config['c3d'].get('use_2d', True):
      model = C3D_2D(
        in_channels=n_input_channels,
        out_channels=out_channels
      )
    else:
      model = C3D(
        in_channels=n_input_channels,
        out_channels=out_channels
      )
  elif 'beamdiff' in model_config:
    beamdiff_config = model_config['beamdiff']
    model = build_beamdiff(beamdiff_config, structure_selector, out_channels=out_channels)
  elif 'dosediff' in model_config:
    dosediff_config = model_config['dosediff']
    model = build_dosediff(dosediff_config, structure_selector, out_channels=out_channels)
  elif 'diffdp' in model_config:
    diffdp_config = model_config['diffdp']
    model = build_diffdp(diffdp_config, structure_selector, out_channels=out_channels)
  elif 'spdiffdose' in model_config:
    spdiffdose_config = model_config['spdiffdose']
    model = build_spdiffdose(spdiffdose_config, structure_selector, out_channels=out_channels)
  elif 'deeplabv3plus' in model_config:
    model = build_deeplabv3plus(model_config, structure_selector, out_channels=out_channels)
  elif 'hdunet' in model_config:
    model = build_hdunet(model_config, structure_selector, out_channels=out_channels)
  elif 'mednext' in model_config:
    model = build_mednext_model(model_config, structure_selector, out_channels=out_channels)
  elif 'analytical_dose' in model_config:
    cfg = model_config.get('analytical_dose', {})
    model = AnalyticalDoseModel(
      structure_selector=structure_selector,
      falloff_constant=cfg.get('falloff_constant', 25.0),
      sparing_radius=cfg.get('sparing_radius', 15.0),
      output_normalized=cfg.get('output_normalized', True),
      output_uncertainty=(out_channels == 2),
      n_samples=cfg.get('n_samples', 16),
      falloff_range=tuple(cfg.get('falloff_range', [20.0, 35.0])),
      sparing_range=tuple(cfg.get('sparing_range', [10.0, 20.0])),
    )
  else:
    model = MambaUnet(
      img_size=model_config['data']['img_size'],
      in_struc_channels=n_input_channels,
      num_layers=model_config['diffusion'].get('arch', {}).get('num_layers', 4),
    )

  return model
