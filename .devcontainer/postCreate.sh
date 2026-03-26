#!/bin/bash
#cd /home/ws/ros2_ws
#mkdir -p src
#sudo rosdep update
#sudo rosdep install --from-paths src --ignore-src -y
#sudo chown -R $(whoami) /home/ws
#colcon build --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON


## AI SLOP -- FIX DONE ONE FRIDAY EVENING
#!/bin/bash
cd /home/ws/ros2_ws
mkdir -p src

sudo apt install python3-scipy -y 

# Remove any duplicate gpd directories
if [ -d "/home/ws/ros2_ws/gpd" ]; then
  rm -rf /home/ws/ros2_ws/gpd
fi

# Only clone gpd if it doesn't exist in src directory
if [ ! -d "/home/ws/ros2_ws/src/gpd" ]; then
  cd /home/ws/ros2_ws/src
  git clone https://github.com/atenpas/gpd.git
fi

cd /home/ws/ros2_ws

# Build and install GPD first (it's a C++ library, not a ROS package)
if [ -d "/home/ws/ros2_ws/src/gpd" ]; then
  cd /home/ws/ros2_ws/src/gpd
  mkdir -p build
  cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=/usr/local
  sudo make -j$(nproc)
  sudo make install
  sudo ldconfig
fi

cd /home/ws/ros2_ws

sudo rosdep update
# Use --skip-keys to ignore missing rosdep keys like scipy
sudo rosdep install --from-paths src --ignore-src --skip-keys scipy -y
sudo chown -R $(whoami) /home/ws
# Now build the ROS packages (excluding gpd since it's already built)
colcon build --packages-skip gpd --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_EXPORT_COMPILE_COMMANDS=ON