#!/usr/bin/env bash

docker build \
	--pull \
	--progress=plain \
	--ssh default \
	-t segmentation-distortion \
	-f Dockerfile .


#	--build-arg user=$USER\
#        --build-arg uid=$UID\

