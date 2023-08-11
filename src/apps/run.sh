!/bin/sh

#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method res_dae --task corr --scanner A --debug no --post base & 
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method res_dae --task corr --scanner B --debug no --post base &
#wait
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method res_dae --task corr --scanner C --debug no --post base &
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method res_dae --task corr --scanner D --debug no --post base &
#wait

#vals="0 1 2 3 4 5 6 7 8 9"
#vals="0"
#for i in $vals; do
#    echo "$i"
#    CUDA_VISIBLE_DEVICES=0 python train_dae_acdc.py --post venus -i $i
#done



CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner A --debug no --post venus_single --save_id single_CE & 
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner B --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner C --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner D --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner val --debug no --post venus_single --save_id single_CE &
wait

CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner A --debug no --post venus --save_id multi_CE & 
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner B --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner C --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner D --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner val --debug no --post venus --save_id multi_CE &
wait


CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner A --debug no --post venus_mse --save_id multi_CEMSE & 
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner B --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner C --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner D --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=1 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner val --debug no --post venus_mse --save_id multi_CEMSE &
wait

CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner A --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner B --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner C --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner D --debug no --post venus_single --save_id single_CE &
CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner val --debug no --post venus_single --save_id single_CE &
wait

CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner A --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner B --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner C --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner D --debug no --post venus --save_id multi_CE &
CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner val --debug no --post venus --save_id multi_CE &
wait

CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner A --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner B --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner C --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner D --debug no --post venus_mse --save_id multi_CEMSE &
CUDA_VISIBLE_DEVICES=1 python eval_pixel.py --n_unets 10 --net_out mms --method multi --scanner val --debug no --post venus_mse --save_id multi_CEMSE &


#



#printf "%s\n" "${vals[@]}"




# CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner val --debug no &
