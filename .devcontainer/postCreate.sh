#!/bin/bash
cd /home/ws/ros2_ws
mkdir -p src
sudo rosdep update
sudo rosdep install --from-paths src --ignore-src -y
sudo chown -R $(whoami) /home/ws
echo 'export PYTHONPATH=$PYTHONPATH:/home/ws/source' >> ~/.bashrc
colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON