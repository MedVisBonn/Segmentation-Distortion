#!/bin/bash

CUDA_DEVICE=1

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_pr.py \
    +dae=channelDAE_config \
    +eval=pixel_config \
    +run.data_key='brain' \
    +run.iteration=0 \
    ++dae.name='channelDAE_up3' \
    ++dae.trainer.disabled_ids="['shortcut0', 'shortcut1', 'shortcut2']" \
    ++debug=false \
    ++eval.data.subset.apply=true \
    ++eval.data.subset.params.n_cases=256 \
    ++eval.data.testing="[1,2,3,4,5]" \
    ++eval.data.validation=true


# ['shortcut0', 'shortcut1', 'shortcut2']
# ['A', 'B', 'C', 'D']
# [1,2,3,4,5]
