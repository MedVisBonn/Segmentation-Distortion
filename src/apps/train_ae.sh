
#!/bin/bash

## Command line arguments
DATA_KEY=$1
CUDA_DEVICE=$2
DEBUG=$3
AE_RESIDUAL=false
## Globals
LOG=true
TRAIN=true
EVAL=true
## Model params
P=0.9
MASK='per_pixel'
AE_POSTFIX='_masked_perpixel_0-9'
# AE_RESIDUAL=false


# for UNET_NAME in 'default-8' 'monai-8-4-4' 'monai-16-4-4' 'default-16' 'monai-32-4-4' 'monai-64-4-4'; do
# for UNET_NAME in 'default-8' 'monai-16-4-4'; do
for UNET_NAME in 'default-8'; do
    IFS=- read -r UNET_ARCH UNET_N_FILTERS_INIT UNET_DEPTH UNET_NUM_RES_UNITS <<< $UNET_NAME

    # for AE_NAME in 'ResDAE-8' 'ResDAE-32' 'ResDAE-128' ; do 'CompressionDAE-bottleneck-3-4'  'ChannelDAE-bottleneck-3-4' 'ChannelDAE-all-3-1' 'ChannelDAE-all-3-1' 'ResDAE-bottleneck-32'
    for AE_NAME in 'ResMAE-bottleneck-128' ; do
        # echo "UNET: $UNET_NAME, DAE: $DAE_NAME"
        IFS=- read -r AE_ARCH PLACEMENT AE_DEPTH AE_BLOCK<<< $AE_NAME

        if [ "$TRAIN" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_ae.py \
                -cn basic_config_venusberg \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                ++unet."$DATA_KEY".pre="$UNET_NAME" \
                ++unet."$DATA_KEY".arch="$UNET_ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$UNET_N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$UNET_DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$UNET_NUM_RES_UNITS" \
                +dae="$AE_ARCH"_config \
                ++dae.name="$AE_NAME" \
                ++dae.postfix="$AE_POSTFIX" \
                ++dae.placement="$PLACEMENT" \
                ++dae.arch.depth="$AE_DEPTH" \
                ++dae.arch.block="$AE_BLOCK" \
                ++dae.arch.residual="$AE_RESIDUAL" \
                ++dae.arch.p="$P" \
                ++dae.arch.mask="$MASK" \
                ++debug="$DEBUG" \
                ++wandb.log="$LOG" \
                ++wandb.name="AE_${DATA_KEY}_${UNET_NAME}_${AE_NAME}${AE_POSTFIX}"
        fi

        if [ "$EVAL" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_pr.py \
                -cn basic_config_venusberg \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                ++data.heart.mnm.selection='all_cases' \
                ++unet."$DATA_KEY".pre="$UNET_NAME" \
                ++unet."$DATA_KEY".arch="$UNET_ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$UNET_N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$UNET_DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$UNET_NUM_RES_UNITS" \
                +dae="$AE_ARCH"_config \
                ++dae.name="$AE_NAME" \
                ++dae.postfix="$AE_POSTFIX" \
                ++dae.placement="$PLACEMENT" \
                ++dae.arch.depth="$AE_DEPTH" \
                ++dae.arch.block="$AE_BLOCK" \
                ++dae.arch.residual="$AE_RESIDUAL" \
                ++debug="$DEBUG" \
                +eval=pixel_config \
                ++eval.data.subset.apply=false \
                ++eval.data.subset.params.n_cases=256 \
                ++eval.data.subset.params.fraction=0.1 \
                ++eval.data.training=false \
                ++eval.data.validation=true \
                ++eval.data.testing="all"
        fi
    done
done