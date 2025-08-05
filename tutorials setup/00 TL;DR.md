# TL;DR

### Robot Setup
- Setup wireless connection with Moonlight
- Prepare robot after turning it on
	- `stretch_free_robot_process.py`
	- `stretch_robot_home.py`
	- `stretch_system_check.py` (optional)
- Shut-down robot with controller
	- `stretch_gamepad_teleop.py`
	- Press Back-Button on controller for 2s
	- Turn off robot body switch

### Project Setup
- Clone project & start as Dev Container in VS Code
- Install models & pull docker images
- Run `urdf_setup.py` to modify robot urdf for IK

### Framework Preprocessing
- Create FUNMAP mapping 
	- `ros2 launch stretch_funmap mapping.launch.py` 
	- `ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True`
- Create low-res robot point cloud
	- `ros2 launch stretch_funmap mapping.launch.py map_yaml:="/home/hello-robot/stretch_user/debug/merged_maps/merged_map_[timestamp]` with the previously generated map
	- `ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True`
	- `ros2 run stretch_package pointcloud_service`
	- `ros2 run stretch_package pointcloud_client` on the workstation
- Create high-res ipad point cloud
	- Save point cloud in scan folder as `mesh.ply` & copy folder to the workstation
- Run `pointcloud_preprocessing.py` to merge & align point clouds for navigation & localization
- Run `scenegraph_preprocessing.py` to create a scene graph from Mask3D results
	
