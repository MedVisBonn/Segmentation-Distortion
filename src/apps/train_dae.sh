
#!/bin/bash

CUDA_DEVICE=1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_dae.py \
    +run.iteration=0 \
    +run.data_key='brain' \
    +dae=channelDAE_config \
    ++dae.name='channelDAE_up3' \
    ++dae.trainer.disabled_ids="['shortcut0', 'shortcut1', 'shortcut2']" \
    ++debug=False \
    ++wandb.log=True \
    ++wandb.name='brain_channelDAE_up3'
