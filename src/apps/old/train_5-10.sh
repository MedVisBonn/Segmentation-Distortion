#!/bin/sh
vals="5 6 7 8 9"
#vals="0"
for i in $vals; do
    echo "$i"
    CUDA_VISIBLE_DEVICES=2 python train_dae_augs.py -i $i
done

