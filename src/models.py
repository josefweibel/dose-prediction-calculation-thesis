from mamba_model import MambaUnet
from mednext import build_mednext_model
from c3d_model import C3D, C3D_2D
from beamdiff_model import build_beamdiff
from spdiffdose_model import build_spdiffdose
from dosediff_model import build_dosediff
from deeplabv3plus_model import build_deeplabv3plus


def build_model(model_config, structure_selector):

  n_input_channels = structure_selector.total_channels

  if 'c3d' in model_config:
    if model_config['c3d'].get('use_2d', True):
      model = C3D_2D(
        in_channels=n_input_channels,
        out_channels=1
      )
    else:
      model = C3D(
        in_channels=n_input_channels,
        out_channels=1
      )
  elif 'beamdiff' in model_config:
    beamdiff_config = model_config['beamdiff']
    model = build_beamdiff(beamdiff_config, structure_selector)
  elif 'spdiffdose' in model_config:
    spdiffdose_config = model_config['spdiffdose']
    model = build_spdiffdose(spdiffdose_config, structure_selector)
  elif 'dosediff' in model_config:
    dosediff_config = model_config['dosediff']
    model = build_dosediff(dosediff_config, structure_selector)
  elif 'deeplabv3plus' in model_config:
    model = build_deeplabv3plus(model_config, structure_selector)
  elif 'mednext' in model_config:
    model = build_mednext_model(
      model_config,
      n_input_channels,
      for_diffusion='diffusion' in model_config,
    )
  else:
    model = MambaUnet(
      img_size=model_config['data']['img_size'],
      in_struc_channels=n_input_channels,
      num_layers=model_config['diffusion'].get('arch', {}).get('num_layers', 4),
    )

  return model
