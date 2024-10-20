
#!/bin/bash

DEBUG=false
LOG=true
CUDA_DEVICE=3

TRAIN=true
EVAL_SCANNER_SHIFT=true
EVAL_ARTIFICIAL_SHIFT=true

POSTFIX='_no-blurring'
AUGMENT='global_without_GaussianBlurTransform'

#'monai-16-4-4' 'brain' 
for DATA_KEY in 'heart'; do
    for NAME in 'default-8'; do

        IFS=- read -r ARCH N_FILTERS_INIT DEPTH NUM_RES_UNITS <<< $NAME

        if [ "$TRAIN" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train_unet.py \
                -cn basic_config_venusberg \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                +run.transform_key="$AUGMENT" \
                ++unet."$DATA_KEY".pre="${NAME}${POSTFIX}" \
                ++unet."$DATA_KEY".arch="$ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
                ++unet."$DATA_KEY".training.num_batches_per_epoch=100 \
                ++unet."$DATA_KEY".training.patience=10 \
                ++unet."$DATA_KEY".training.epochs=250 \
                ++debug=$DEBUG \
                ++wandb.log=$LOG \
                ++wandb.name="${DATA_KEY}_unet_${NAME}"
        fi

        if [ "$EVAL_SCANNER_SHIFT" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_unet_scanner_shift.py \
                -cn basic_config_venusberg \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                +run.transform_key="$AUGMENT" \
                +eval=unet_config \
                ++eval.data.training=true \
                ++eval.data.validation=true \
                ++eval.data.testing='all' \
                ++unet."$DATA_KEY".pre="${NAME}${POSTFIX}" \
                ++unet."$DATA_KEY".arch="$ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
                ++debug=$DEBUG
        fi

        if [ "$EVAL_ARTIFICIAL_SHIFT" = true ]; then
            CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python eval_unet_artificial_shift.py \
                -cn basic_config_venusberg \
                +run.iteration=0 \
                +run.data_key="$DATA_KEY" \
                +run.transform_key="$AUGMENT" \
                +eval=unet_config \
                ++eval.data.training=true \
                ++eval.data.validation=true \
                ++eval.data.testing='all' \
                ++unet."$DATA_KEY".pre="${NAME}${POSTFIX}" \
                ++unet."$DATA_KEY".arch="$ARCH" \
                ++unet."$DATA_KEY".n_filters_init="$N_FILTERS_INIT" \
                ++unet."$DATA_KEY".depth="$DEPTH" \
                ++unet."$DATA_KEY".num_res_units="$NUM_RES_UNITS" \
                ++debug=$DEBUG
        fi
    done
done


# for DATA_KEY in 'brain' 'heart'; do
    # for NAME in 'monai-16-4-8' 'monai-32-4-4' 'monai-64-4-4' 'swinunetr'; do

# for DATA_KEY in 'brain' 'heart'; do
#     for NAME in 'default-8' 'default-16' 'monai-16-4-4'; do
# "['A', 'B', 'C', 'D']"
# "[1,2,3,4,5]"
#default_16-4-4