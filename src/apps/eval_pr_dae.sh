#!/bin/bash

CUDA_VISIBLE_DEVICES=2 python eval_pr.py \
    +dae=unetDAE_config \
    +eval=pixel_config \
    +run.data_key='brain' \
    +run.iteration=3 \
    ++dae.name='calgary_unet_res' \
    ++dae.trainer.disabled_ids='[]' \
    ++debug=false \
    ++eval.data.subset.apply=true \
    ++eval.data.subset.params.n_cases=256 \
    ++eval.data.testing='[1,2,3,4,5]' \
    ++eval.data.validation=true
