ARG DOCKER_REPO=osrf/ros
ARG ROS_DISTRO=humble
ARG IMAGE_SUFFIX=-desktop-full
FROM $DOCKER_REPO:$ROS_DISTRO$IMAGE_SUFFIX

# ──────────────────────────── [NEW] KEY FIX ──────────────────────────────────
# The ROS 2 signing key that ships in images built before 2025‑06‑01 is
# expired.  Refresh it *before* the first apt-get update to avoid “EXPKEYSIG”.
RUN set -eux; \
    apt-get update || true;                                           \
    apt-get install -y --no-install-recommends curl gnupg;            \
    rm -f /usr/share/keyrings/ros-archive-keyring.gpg || true;        \
    curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
         -o /usr/share/keyrings/ros-archive-keyring.gpg
# ────────────────────────── [END KEY FIX] ────────────────────────────────────

RUN set -eux; \
    apt-get update && \
    apt-get install -y --no-install-recommends sudo && \
    rm -rf /var/lib/apt/lists/*

ARG USERNAME
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Create the user
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    #&& apt-get update \
    #&& apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install additonal packages - add any that you need
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python-is-python3 ssh neovim git ros-humble-control-msgs ros-humble-navigation2
ENV SHELL=/bin/bash

RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y python3-pip python-is-python3 ssh neovim git

RUN apt-get update && apt-get install -y \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool \
    python3-pyqt5.qtsvg \
    python3-pyqt5.qtwebsockets \
    python3-pyqt5.qtmultimedia \
    python3-pyqt5.qtwebchannel

RUN apt-get update && apt-get install -y \
    ros-$ROS_DISTRO-controller-manager

RUN apt-get update && apt-get install -y \
    ros-humble-controller-manager \
    ros-humble-moveit \
    ros-humble-moveit-common \
    ros-humble-moveit-ros-planning \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-ros-visualization \
    ros-humble-moveit-ros-perception \
    ros-humble-moveit-setup-assistant

RUN apt-get update && apt-get install -y \
    ros-humble-nav2-bringup \
    ros-humble-rviz2 \
    ros-humble-joint-state-publisher-gui \
    ros-humble-xacro \
    ros-humble-rqt \
    ros-humble-rqt-graph \
    ros-humble-rqt-joint-trajectory-controller

RUN apt-get update && apt-get install -y \
    ros-humble-controller-manager \
    ros-humble-joint-state-broadcaster \
    ros-humble-joint-trajectory-controller \
    ros-humble-diff-drive-controller \
    ros-humble-robot-state-publisher \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-hardware-interface \
    ros-humble-transmission-interface

RUN apt-get update && apt-get install -y \
    ros-humble-teleop-twist-keyboard \
    ros-humble-tf2-tools \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-tf2-sensor-msgs \
    ros-humble-tf2-eigen \
    ros-humble-tf2-py 

# RUN apt-get update && apt-get install -y \
#     pip install pyoptas

RUN apt-get update && apt-get install -y ros-humble-desktop ros-dev-tools rviz python3-pip

# Install python dependencies
RUN pip install open3d openai ikpy graphviz ultralytics apriltag git+https://github.com/openai/CLIP.git "numpy<1.25" transformers hydra-core



# Source ROS environment automatically
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc
RUN echo "source /home/ws/ros2_ws/install/setup.bash" >> /home/$USERNAME/.bashrc

# Set the default user
USER $USERNAME
CMD ["/bin/bash"]
