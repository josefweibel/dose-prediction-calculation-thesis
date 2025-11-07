def build_mednext_model(model_config, in_channels):
  from nnunet_mednext import MedNeXt

  config = model_config['mednext']

  # Medium
  return MedNeXt(
    in_channels=in_channels,
    n_channels=32,
    n_classes=1,
    exp_r=[2,3,4,4,4,4,4,3,2],
    kernel_size=config['kernel_size'],
    deep_supervision=config['deep_supervision'],
    do_res=True,
    do_res_up_down=True,
    block_counts=[3,4,4,4,4,4,4,4,3],
    checkpoint_style='outside_block',
    dim='2d',
  )
