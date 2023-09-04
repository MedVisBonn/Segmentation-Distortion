#!/usr/bin/env bash

docker build \
	--pull \
	--progress=plain \
	--ssh default \
	-t segmentation-distortion/replicate:1.0 \
	-f Dockerfile .


#	--build-arg user=$USER\
#        --build-arg uid=$UID\

