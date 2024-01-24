#!/bin/sh
vals="0 1 2 3 4 5 6 7 8 9"

for i in $vals; do
    echo "$i"
    CUDA_VISIBLE_DEVICES=2 python train_dae_augs.py -i $i --residual 1
done

### - Evaluation

## Pixel Level
CUDA_VISIBLE_DEVICES=2 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner A --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner B --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner C --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner D --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner val --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
wait

## Image Level
CUDA_VISIBLE_DEVICES=2 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner A --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same & 
CUDA_VISIBLE_DEVICES=2 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner B --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner C --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner D --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=2 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner val --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &

