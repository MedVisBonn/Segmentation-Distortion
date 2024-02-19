
#!/bin/bash

DEBUG=false
LOG=true
TRAIN=true
EVAL=true
CUDA_DEVICE=1
# DATA_KEY='heart'
# UNET_NAME='default-8'
DAE_NAME='heart_resDAE_testing'


# unet archs
## datasets
### dae archs
# for UNET_NAME in 'default-8' 'default-16' 'monai-8-4-4' 'monai-16-4-4' 'monai-32-4-4' 'monai-64-4-4'; do
#     for DATA_KEY in 'brain' 'heart'; do

IFS=- read -r ARCH N_FILTERS_INIT DEPTH NUM_RES_UNITS <<< $UNET_NAME
if [ "$TRAIN" = true ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_dae.py \
        +run.iteration=0 \
        +run.data_key="$DATA_KEY" \
        ++unet."$DATA_KEY".pre="$UNET_NAME" \
        ++unet."$DATA_KEY".arch="$ARCH" \
        ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
        ++unet."$DATA_KEY".depth="$DEPTH" \
        ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
        +dae=resDAE_config \
        ++dae.name="$DAE_NAME" \
        ++dae.arch.depth=20 \
        ++dae.identity_swivels="[0,1,2]" \
        ++debug="$DEBUG" \
        ++wandb.log="$LOG" \
        ++wandb.name='resDAE_testing'
fi

if [ "$EVAL" = true ]; then
    CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_pr.py \
        +run.iteration=0 \
        +run.data_key="$DATA_KEY" \
        ++unet."$DATA_KEY".pre="$UNET_NAME" \
        ++unet."$DATA_KEY".arch="$ARCH" \
        ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
        ++unet."$DATA_KEY".depth="$DEPTH" \
        ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
        +dae=resDAE_config \
        ++dae.name="$DAE_NAME" \
        ++dae.arch.depth=20 \
        ++dae.identity_swivels="[0,1,2]" \
        ++debug="$DEBUG" \
        +eval=pixel_config \
        ++eval.data.subset.apply=true \
        ++eval.data.subset.params.n_cases=256 \
        ++eval.data.training=false \
        ++eval.data.validation=true \
        ++eval.data.testing="all"
fi

#     done
# done
