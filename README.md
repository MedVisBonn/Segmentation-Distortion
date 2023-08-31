# Segmentation Distortion
Here you will soon find the step-by-step instructions to install and work with the methodology presented in our paper.

## Setup

### Git
To access the code for your own work, simply clone the repository and take what you need
```bash
git clone git@github.com:MedVisBonn/Segmentation-Distortion.git
```
### Docker
The easiest way to replicate the results of the paper is via Docker. You can find all necessary template files in the [docker directory](https://github.com/MedVisBonn/Segmentation-Distortion/tree/main/docker). To build the image, copy the Dockerfile and build.sh and run
```bash 
bash build.sh -t . TODO
```
Alternatively, you can download a pre-build image from docker-hub
```bash
TODO
```
Once the image is ready, modify the run.sh to mount datasets and run a container
```bash
bash run.sh
```
You should be attached to an interactive session within the container automatically. 


## Requirements
The docker image is build on top of [NVidia's PyTorch image 23.07](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-07.html#rel-23-07) and thus needs NVidia drivers 530 or later. Additionally, to run GPU accelerated containers, the nvidia-container-toolkit has to be installed installed and configured. For more information, check out the official [NVIDIA Container Toolkit documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
Package dependencies are handled within the docker image.

## Usage
To explore the project, you can run a jupyter lab server on the port specified in the [Dockerfile](https://github.com/MedVisBonn/Segmentation-Distortion/blob/main/docker/Dockerfile).
```bash
jupyter lab --no-browser --allow-root --port 8888
```
## License

## Citation
TBA
## Contact

For any questions or clarifications, feel free to reach out

lennartz (ät) cs.uni-bonn.de

(Please allow a couple of days for a response)
