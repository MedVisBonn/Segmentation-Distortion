
#!/bin/bash

DEBUG=false
LOG=true
TRAIN=true
EVAL=true
CUDA_DEVICE=1
DATA_KEY='brain'
NAME='default-8'


IFS=- read -r ARCH N_FILTERS_INIT DEPTH NUM_RES_UNITS <<< $NAME
if [ "$TRAIN" = true ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_dae.py \
        +run.iteration=0 \
        +run.data_key="$DATA_KEY" \
        ++unet."$DATA_KEY".pre="$NAME" \
        ++unet."$DATA_KEY".arch="$ARCH" \
        ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
        ++unet."$DATA_KEY".depth="$DEPTH" \
        ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
        +dae=resDAE_config \
        ++dae.name='resDAE_testing' \
        ++dae.arch.depth=20 \
        ++dae.identity_swivels="[0,1,2]" \
        ++debug=$DEBUG \
        ++wandb.log=$LOG \
        ++wandb.name='resDAE_testing'
fi

if [ "$EVAL" = true ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_pr.py \
        +run.iteration=0 \
        +run.data_key="$DATA_KEY" \
        ++unet."$DATA_KEY".pre="$NAME" \
        ++unet."$DATA_KEY".arch="$ARCH" \
        ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
        ++unet."$DATA_KEY".depth="$DEPTH" \
        ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
        +dae=resDAE_config \
        ++dae.name='resDAE_testing' \
        ++dae.arch.depth=20 \
        ++dae.identity_swivels="[0,1,2]" \
        ++debug=$DEBUG \
        +eval=pixel_config \
        ++eval.data.subset.apply=true \
        ++eval.data.subset.params.n_cases=256 \
        ++eval.data.training=false \
        ++eval.data.validation=true \
        ++eval.data.testing="[1,2,3,4,5]" \
fi