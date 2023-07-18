#!/bin/sh

CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner A --debug no &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner B --debug no &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner C --debug no &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner D --debug no &
CUDA_VISIBLE_DEVICES=3 python eval_image.py --net_out mms --method ensemble --task ood --scanner val --debug no &