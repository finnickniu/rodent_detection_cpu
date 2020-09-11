
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"
RUN conda install pytorch torchvision cpuonly -c pytorch
RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
run pip install opencv-python pillow scikit-image
# Install MMDetection
RUN conda clean --all
RUN git clone https://github.com/finnickniu/rodent_detection_cpu.git /rodent_detection_cpu
WORKDIR /rodent_detection_cpu
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
