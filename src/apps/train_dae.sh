
#!/bin/bash

## Command line arguments
DATA_KEY=$1
CUDA_DEVICE=$2
DEBUG=$3
## Globals
LOG=true
TRAIN=true
EVAL=true

# for UNET_NAME in 'default-8' 'monai-8-4-4' 'monai-16-4-4' 'default-16' 'monai-32-4-4' 'monai-64-4-4'; do
# for UNET_NAME in 'default-8' 'monai-16-4-4'; do
for UNET_NAME in 'monai-16-4-4' ; do
    IFS=- read -r UNET_ARCH UNET_N_FILTERS_INIT UNET_DEPTH UNET_NUM_RES_UNITS <<< $UNET_NAME

    # for DAE_NAME in 'ResDAE-8' 'ResDAE-32' 'ResDAE-128' ; do 'ChannelDAE-bottleneck-3-4' 'ChannelDAE-all-3-1' 'ChannelDAE-all-3-1' 'ResDAE-bottleneck-32'
    for DAE_NAME in 'ChannelDAE-bottleneck-3-4' 'ResDAE-all-32'; do
        # echo "UNET: $UNET_NAME, DAE: $DAE_NAME"
        IFS=- read -r DAE_ARCH PLACEMENT DAE_DEPTH DAE_BLOCK<<< $DAE_NAME

        if [ "$TRAIN" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_dae.py \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                ++unet."$DATA_KEY".pre="$UNET_NAME" \
                ++unet."$DATA_KEY".arch="$UNET_ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$UNET_N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$UNET_DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$UNET_NUM_RES_UNITS" \
                +dae="$DAE_ARCH"_config \
                ++dae.name="$DAE_NAME" \
                ++dae.placement="$PLACEMENT" \
                ++dae.arch.depth="$DAE_DEPTH" \
                ++dae.arch.block="$DAE_BLOCK" \
                ++debug="$DEBUG" \
                ++wandb.log="$LOG" \
                ++wandb.name="DAE_${DATA_KEY}_${UNET_NAME}_${DAE_NAME}"
        fi

        if [ "$EVAL" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_pr.py \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                ++unet."$DATA_KEY".pre="$UNET_NAME" \
                ++unet."$DATA_KEY".arch="$UNET_ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$UNET_N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$UNET_DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$UNET_NUM_RES_UNITS" \
                +dae="$DAE_ARCH"_config \
                ++dae.name="$DAE_NAME" \
                ++dae.placement="$PLACEMENT" \
                ++dae.arch.depth="$DAE_DEPTH" \
                ++dae.arch.block="$DAE_BLOCK" \
                ++debug="$DEBUG" \
                +eval=pixel_config \
                ++eval.data.subset.apply=true \
                ++eval.data.subset.params.n_cases=256 \
                ++eval.data.training=false \
                ++eval.data.validation=true \
                ++eval.data.testing="all"
        fi
    done
done