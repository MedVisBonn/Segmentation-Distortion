
#!/bin/bash

## Command line arguments
CUDA_DEVICE=$1
DEBUG=$2

## Globals
TESTING='all'
# TESTING='['A']'

DETECTOR='pooling'
POOL='none'
SIGMA_ALGO='diagonal'
DIST_FN='squared_mahalanobis'
# UMAP='diff_grad'
UMAP='diff_grad'

LR=2e-1
SWIVELS='all'
# POSTFIX='all_propagated_perpixel_diffgrad'
POSTFIX='perpixel_squared-mahalanobis'


# for UNET_NAME in 'default-8' 'monai-8-4-4' 'monai-16-4-4' 'default-16' 'monai-32-4-4' 'monai-64-4-4'; do
# for UNET_NAME in 'default-8' 'monai-16-4-4'; do
# for UNET_NAME in 'default-8' 'monai-16-4-4' ; do
for UNET_NAME in 'default-8' ; do
    IFS=- read -r UNET_ARCH UNET_N_FILTERS_INIT UNET_DEPTH UNET_NUM_RES_UNITS <<< $UNET_NAME

    # for DAE_NAME in 'ResDAE-8' 'ResDAE-32' 'ResDAE-128' ; do 'CompressionDAE-bottleneck-3-4'  'ChannelDAE-bottleneck-3-4' 'ChannelDAE-all-3-1' 'ChannelDAE-all-3-1' 'ResDAE-bottleneck-32'
    for DATA_KEY in 'heart'; do

        CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_mahalanobis.py \
            -cn basic_config_venusberg \
            +run.iteration=0 \
            +run.data_key="$DATA_KEY" \
            +run.swivels="$SWIVELS" \
            ++unet."$DATA_KEY".pre="$UNET_NAME" \
            ++unet."$DATA_KEY".arch="$UNET_ARCH" \
            ++unet."$DATA_KEY".n_filters_init="$UNET_N_FILTERS_INIT" \
            ++unet."$DATA_KEY".depth="$UNET_DEPTH" \
            ++unet."$DATA_KEY".num_res_units="$UNET_NUM_RES_UNITS" \
            ++debug="$DEBUG" \
            +eval=mahalanobis_config \
            ++eval.logging.postfix="$POSTFIX" \
            ++eval.params.detector="$DETECTOR" \
            ++eval.params.pool="$POOL" \
            ++eval.params.sigma_algorithm="$SIGMA_ALGO" \
            ++eval.params.dist_fn="$DIST_FN" \
            ++eval.params.lr="$LR" \
            ++eval.params.umap="$UMAP" \
            ++eval.data.testing="$TESTING"
    done
done