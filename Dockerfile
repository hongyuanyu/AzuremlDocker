# Tag: nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
# FROM mcr.microsoft.com/azureml/base-gpu:latest
FROM mcr.microsoft.com/azureml/base-gpu:intelmpi2018.3-cuda9.0-cudnn7-ubuntu16.04
ENV STAGE_DIR=/root/gpu/install 
RUN mkdir -p $STAGE_DIR


# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        --allow-change-held-packages \
        build-essential \
        cmake \
        wget \
        vim \
        tmux \
        htop \
        git \
        unzip \
        libnccl2 \
        libnccl-dev \
        ca-certificates \
        libjpeg-dev

# Install lib for video
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository -y ppa:jonathonf/ffmpeg-3
RUN apt update && apt-get install -y libavformat-dev libavcodec-dev libswscale-dev libavutil-dev libswresample-dev
RUN apt-get install -y ffmpeg
RUN export LIBRARY_PATH=/usr/local/lib:$LIBRARY_PATH

# Set timezone
RUN ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime

# Get Conda-ified Python.
#RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
#    wget --quiet https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh -O ~/anaconda.sh && \
#    sh ~/anaconda.sh -b -p /opt/conda && \
#    rm ~/anaconda.sh

ENV PATH /opt/miniconda/bin:$PATH
# Install general libraries
RUN conda install -y python=3.6 numpy pyyaml scipy ipython mkl scikit-learn matplotlib pandas setuptools Cython h5py graphviz
RUN conda clean -ya
RUN conda install -y mkl-include cmake cffi typing cython
RUN conda install -y -c mingfeima mkldnn
RUN pip install boto3 addict tqdm regex pyyaml opencv-python azureml-defaults


# Set CUDA_ROOT
RUN export CUDA_HOME="/usr/local/cuda"

# Install pytorch
RUN conda install -y pytorch torchvision  -c pytorch



# Install horovod
RUN HOROVOD_GPU_ALLREDUCE=NCCL pip install --no-cache-dir horovod==0.15.2


# Install apex
WORKDIR $STAGE_DIR
RUN pip uninstall -y apex || :
# SHA is something the user can touch to force recreation of this Docker layer,
# and therefore force cloning of the latest version of Apex
RUN SHA=ToUcHMe git clone https://github.com/NVIDIA/apex.git
WORKDIR $STAGE_DIR/apex
RUN python setup.py install --cuda_ext --cpp_ext
WORKDIR $STAGE_DIR
RUN rm -rf apex
