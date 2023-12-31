FROM nvcr.io/nvidia/pytorch:23.07-py3

# ARG user
# ARG uid

### - install standard capabilities
RUN apt-get update
RUN apt-get install -y \
	git \
	vim \
	binutils \
	tmux

## try conda
#RUN ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
#RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> /.bashrc
#
#RUN cat /.bashrc
#
##ENV PATH /opt/conda/bin:$PATH
#RUN conda init bash
#SHELL ["conda", "run", "--no-capture-output",  "/bin/bash", "-c"]
#RUN conda install matplotlib

### - PIP
RUN pip install wandb deep-pipe scikit-learn torchmetrics tqdm 

### - GIT
## setup
# move nvidia-stuff to separate folder
WORKDIR /workspace
RUN mv * nvidia | mkdir nvidia
# set working directory to designated repo dire
WORKDIR /workspace/repos
## segmentation distortion
RUN git clone https://github.com/MedVisBonn/Segmentation-Distortion.git
## surface distances
RUN git clone https://github.com/deepmind/surface-distance.git
RUN pip install surface-distance/
## nnUNet
RUN git clone --branch nnunetv1 https://github.com/MIC-DKFZ/nnUNet.git
WORKDIR /workspace/repos/nnUNet
RUN pip install -e .
# set up variables for nnUnet
WORKDIR /
RUN mkdir -p /data/nnUNet_raw
ENV nnUNet_raw="/data/nnUNet_raw"
RUN mkdir -p /data/nnUNet_preprocessed
ENV nnUNet_preprocessed="/data/nnUNet_preprocessed"
RUN mkdir -p /results/nnUNet_results
ENV nnUNet_results="/results/nnUNet_results"

### - CLEANUP
RUN apt-get autoremove -y && rm -rf /tmp/* /var/tmp/* && apt-get clean && rm -rf /var/lib/apt/lists/*

### - Make user and give rights for workdir and folders we might need access to.
#RUN mkdir /out
#RUN useradd -u ${uid} -ms /bin/bash ${user}
#RUN chown -R ${user} /workspace /out /data /results

### - expose ports for various services
# Make SSH available
EXPOSE 22
EXPOSE 42421
# TensorBoard https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 6006
# IPython https://github.com/tensorflow/tensorflow/blob/master/tensorflow/tools/docker/Dockerfile.gpu
EXPOSE 8889
### - start interactive console on attach
ENV DEBIAN_FRONTEND teletype
