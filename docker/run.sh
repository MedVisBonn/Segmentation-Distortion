#!/usr/bin/env bash

# --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=4 \


docker run \
	-it \
	--net=host \
	--runtime=nvidia \
    --gpus all \
    --privileged \
	--ipc=host \
	--mount type=bind,source="/home/lennartz/data/conp-dataset",target=/data/conp-dataset \
	--mount type=bind,source="/home/lennartz/data/nnUNet_preprocessed",target=/data/nnUNet_preprocessed \
	--mount type=bind,source="/home/lennartz/docker-projects/segmentation-distortion/out",target=/workspace/out \
	segmentation-distortion
	#96740278c511

	# -v "$SSH_AUTH_SOCK:$SSH_AUTH_SOCK" -e SSH_AUTH_SOCK=$SSH_AUTH_SOCK \

# miccai23_reboot_commit

