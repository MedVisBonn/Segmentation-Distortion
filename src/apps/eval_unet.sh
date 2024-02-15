#!/bin/bash

CUDA_DEVICE=1
DATA_KEY='brain'
DEBUG=false

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_unet.py \
    +run.iteration=0 \
    +run.data_key="$DATA_KEY" \
    +eval=unet_config \
    ++eval.data.training=true \
    ++eval.data.validation=true \
    ++eval.data.testing="[1,2,3,4,5]" \
    ++unet."$DATA_KEY".pre='monai_16-4-4' \
    ++unet."$DATA_KEY".arch='monai' \
    ++unet."$DATA_KEY".n_filters_init=16 \
    ++unet."$DATA_KEY".depth=4 \
    ++unet."$DATA_KEY".num_res_units=4 \
    ++debug=$DEBUG
