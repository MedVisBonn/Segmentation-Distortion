#!/bin/sh
vals="0 1 2 3 4 5 6 7 8 9"
#vals="0"
for i in $vals; do
    echo "$i"
    CUDA_VISIBLE_DEVICES=3 python train_dae_acdc.py --post single_MSEMSE_1prior -i $i
done

