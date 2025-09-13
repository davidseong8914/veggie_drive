# Legged Controls Dockerfile (CUDA 12.8.1 + ROS 2 Jazzy + PyTorch 2.2 cu121)
FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=${USER_UID}
ARG CMAKE_VERSION=4.0.0
ARG NUM_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    SHELL=/bin/bash \
    LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8 LANGUAGE=en_US.UTF-8 \
    NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-all} \
    NVIDIA_DRIVER_CAPABILITIES=${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Create user (sudo)
RUN if id -u ${USER_UID} >/dev/null 2>&1; then userdel -r "$(id -un ${USER_UID})" || true; fi && \
    groupadd --gid ${USER_GID} ${USERNAME} && \
    useradd --uid ${USER_UID} --gid ${USER_GID} -m ${USERNAME} && \
    apt-get update && apt-get install -y sudo && \
    echo "${USERNAME} ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME}

# Base packages
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
      build-essential autoconf automake libtool pkg-config \
      ca-certificates locales locales-all \
      python3-full python3-pip \
      wget curl git vim tmux htop less \
      software-properties-common cmake && \
    rm -rf /var/lib/apt/lists/*

# CMake (4.0.0)
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh && \
    bash cmake-${CMAKE_VERSION}-linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm -rf /tmp/*

# -----------------------
# ROS 2 Jazzy (Ubuntu 24.04 / noble)
# -----------------------
SHELL ["/bin/bash", "-lc"]

RUN add-apt-repository universe && \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu noble main" \
      > /etc/apt/sources.list.d/ros2.list && \
    apt-get update && apt-get install -y --no-install-recommends \
      ros-jazzy-desktop \
      python3-rosdep \
      python3-colcon-common-extensions && \
    rm -rf /var/lib/apt/lists/*

RUN rosdep init || true && rosdep update

RUN echo 'source /opt/ros/jazzy/setup.bash' >> /etc/bash.bashrc

# -----------------------
# PyTorch 2.2.x (CUDA 12.1 wheels)
# -----------------------
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir \
      torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 \
      --index-url https://download.pytorch.org/whl/cu121

# Quick checks (non-fatal)
RUN nvcc --version || true

RUN python3 -c "import torch; print('torch', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('torch cuda runtime:', torch.version.cuda)"

RUN bash -lc 'source /opt/ros/jazzy/setup.bash && ros2 --version || true'

# -----------------------
# Livox SDK2
# -----------------------
WORKDIR /root

RUN git clone https://github.com/liyufan/Livox-SDK2.git && \
    cmake -S Livox-SDK2 -B Livox-SDK2/build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && \
    cmake --build Livox-SDK2/build -j${NUM_JOBS} && \
    cmake --install Livox-SDK2/build

# Workspace placeholder (optional)
WORKDIR /root/ros2_ws

# Entrypoint to load ROS env each shell
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod 755 /ros_entrypoint.sh

ENTRYPOINT ["/ros_entrypoint.sh"]
CMD ["bash"]