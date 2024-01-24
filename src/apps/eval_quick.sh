#!/bin/sh


# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 1 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 2 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 3 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 4 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 5 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method single --scanner 6 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
# wait 

# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 1 --debug no --post base --save_id base &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 2 --debug no --post base --save_id base &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 3 --debug no --post base --save_id base &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 4 --debug no --post base --save_id base &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 5 --debug no --post base --save_id base &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method base --scanner 6 --debug no --post base --save_id base &
# wait

# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 1 --debug no --post ensemble --save_id ensemble &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 2 --debug no --post ensemble --save_id ensemble &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 3 --debug no --post ensemble --save_id ensemble &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 4 --debug no --post ensemble --save_id ensemble &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 5 --debug no --post ensemble --save_id ensemble &
# CUDA_VISIBLE_DEVICES=3 python eval_pixel.py --n_unets 10 --net_out calgary --method ensemble --scanner 6 --debug no --post ensemble --save_id ensemble &
# wait


#CUDA_VISIBLE_DEVICES=0 python eval_downstream_heart.py

CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 1 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same & 
CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 2 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 3 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 4 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 5 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out calgary --method single --task corr --scanner 6 --debug no --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same &
wait




# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner A --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same_nNflips & 
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner B --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner C --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner D --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner val --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id localAug_multiImgSingleView_res_balanced_same_nNflips &
# wait

# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method ae --task corr --scanner A --debug no --selection all_cases --post none --save_id ac_MSE &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method ae --task corr --scanner B --debug no --selection all_cases --post none --save_id ac_MSE &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method ae --task corr --scanner C --debug no --selection all_cases --post none --save_id ac_MSE &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method ae --task corr --scanner D --debug no --selection all_cases --post none --save_id ac_MSE &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method ae --task corr --scanner val --debug no --selection all_cases --post none --save_id ac_MSE &
# wait

# CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method gonzales --task corr --scanner A --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id gonzales_acc_nonempty & 
# CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method gonzales --task corr --scanner B --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id gonzales_acc_nonempty &
# CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method gonzales --task corr --scanner C --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id gonzales_acc_nonempty &
# CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method gonzales --task corr --scanner D --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id gonzales_acc_nonempty &
# CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method gonzales --task corr --scanner val --debug no --selection all_cases --post localAug_multiImgSingleView_res_balanced_same --save_id gonzales_acc_nonempty &
# #wait

# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner A --debug no --selection all_cases --post localAug_multiImgSingleView_recon_balanced_same --save_id localAug_multiImgSingleView_recon_balanced_same_nNflips & 
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner B --debug no --selection all_cases --post localAug_multiImgSingleView_recon_balanced_same --save_id localAug_multiImgSingleView_recon_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner C --debug no --selection all_cases --post localAug_multiImgSingleView_recon_balanced_same --save_id localAug_multiImgSingleView_recon_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner D --debug no --selection all_cases --post localAug_multiImgSingleView_recon_balanced_same --save_id localAug_multiImgSingleView_recon_balanced_same_nNflips &
# CUDA_VISIBLE_DEVICES=3 python eval_image.py --n_unets 10 --net_out mms --method single --task corr --scanner val --debug no --selection all_cases --post localAug_multiImgSingleView_recon_balanced_same --save_id localAug_multiImgSingleView_recon_balanced_same_nNflips &