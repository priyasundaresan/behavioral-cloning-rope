ARG cuda_version=9.0
ARG cudnn_version=7
FROM nvidia/cuda:${cuda_version}-cudnn${cudnn_version}-devel

# Install system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
      git \
      libgl1-mesa-glx \
      libhdf5-dev \
      libglib2.0-0 \
      openmpi-bin \
      wget && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-utils git curl vim wget \
    build-essential cmake \
    libopenblas-dev

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && apt-get install -y python3.6 python3.6-dev python3-pip

RUN ln -sfn /usr/bin/python3.6 /usr/bin/python3 && ln -sfn /usr/bin/python3 /usr/bin/python && ln -sfn /usr/bin/pip3 /usr/bin/pip

RUN pip3 install --no-cache-dir --upgrade pip setuptools && \
    echo "alias python='python3'" >> /root/.bash_aliases && \
    echo "alias pip='pip3'" >> /root/.bash_aliases

RUN pip install --upgrade pip && \
    pip install \
        numpy==1.16.4 \
        tensorflow-gpu==1.6.0 \
        keras==2.1.5 \
        opencv_python==3.4.2.17  \
        h5py

RUN apt-get install -y libsm6 libxext6 libxrender-dev
RUN apt install -y --allow-downgrades --allow-change-held-packages libcudnn7-dev=7.0.5.15-1+cuda9.1 libcudnn7=7.0.5.15-1+cuda9.1

WORKDIR /host
CMD ["/bin/bash"]
