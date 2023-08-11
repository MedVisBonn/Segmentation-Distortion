#!/bin/sh


CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner A --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner B --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner C --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner D --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method single --scanner val --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior
wait
#CUDA_VISIBLE_DEVICES=0 python eval_downstream_heart.py

#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner A --debug no --post multi_CEMSE_1prior --save_id multi_CEMSE_1prior & 
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner B --debug no --post multi_CEMSE_1prior --save_id multi_CEMSE_1prior &
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner C --debug no --post multi_CEMSE_1prior --save_id multi_CEMSE_1prior &
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner D --debug no --post multi_CEMSE_1prior --save_id multi_CEMSE_1prior &
#CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method multi --task both --scanner val --debug no --post multi_CEMSE_1prior --save_id multi_CEMSE_1prior &
#wait


#CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method ae --scanner A --debug no --post none --save_id ae_paper &
#CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method ae --scanner B --debug no --post none --save_id ae_paper &
#CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method ae --scanner C --debug no --post none --save_id ae_paper &
#CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method ae --scanner D --debug no --post none --save_id ae_paper &
#CUDA_VISIBLE_DEVICES=0 python eval_pixel.py --n_unets 10 --net_out mms --method ae --scanner val --debug no --post none --save_id ae_paper


CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner A --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior & 
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner B --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner C --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner D --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
CUDA_VISIBLE_DEVICES=0 python eval_image.py --n_unets 10 --net_out mms --method single --task both --scanner val --debug no --post single_MSEMSE_1prior --save_id single_MSEMSE_1prior &
#wait

