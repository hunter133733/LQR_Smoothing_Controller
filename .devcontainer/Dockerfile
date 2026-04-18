FROM althack/ros2:humble-dev 

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    nano \
    tmux \
    libtinfo5 \
    ros-humble-navigation2 \
    ros-humble-nav2-bringup \ 
    ros-humble-turtlebot3-msgs \
    ros-humble-turtlebot3 \
    ros-humble-turtlebot3-gazebo \
    ros-humble-rmw-cyclonedds-cpp \
    && rm -rf /var/lib/apt/lists/*
ENV DEBIAN_FRONTEND=dialog

# Additional pip packages
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir --timeout 2000 -r /tmp/requirements.txt


# Set up auto-source of workspace for ros user
ARG WORKSPACE
RUN echo "if [ -f ${WORKSPACE}/install/setup.bash ]; then source ${WORKSPACE}/install/setup.bash; fi" >> /home/ros/.bashrc

# Set env variables for Nav2
RUN echo "export TURTLEBOT3_MODEL=burger" >> /home/ros/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:/opt/ros/humble/share/turtlebot3_gazebo/models" >> /home/ros/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=\$GAZEBO_MODEL_PATH:${WORKSPACE}/install/mpc/share/mpc/models" >> /home/ros/.bashrc
RUN echo "export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp" >> /home/ros/.bashrc
RUN echo "export CYCLONEDDS_URI=${WORKSPACE}/setup/cyclonedds_lo.xml" >> /home/ros/.bashrc
RUN echo "export ROS_DOMAIN_ID=8" >> /home/ros/.bashrc
