from mamba_model import MambaUnet
from mednext import build_mednext_model
from c3d_model import C3D, C3D_2D


def build_model(model_config, dataset):
  if 'c3d' in model_config:
    # Check if we should use 2D or 3D version
    use_2d = model_config['c3d'].get('use_2d', True)  # Default to 2D for compatibility

    if use_2d:
      model = C3D_2D(
        in_channels=dataset.n_structures,
        out_channels=1
      )
    else:
      model = C3D(
        in_channels=dataset.n_structures,
        out_channels=1
      )
  elif 'mednext' in model_config:
    model = build_mednext_model(
      model_config,
      dataset.n_structures,
      for_diffusion='diffusion' in model_config,
    )
  else:
    model = MambaUnet(
      img_size=model_config['data']['img_size'],
      in_struc_channels=dataset.n_structures,
      num_layers=model_config['diffusion'].get('arch', {}).get('num_layers', 4),
    )

  return model
