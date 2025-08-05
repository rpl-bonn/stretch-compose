# Stretch-Compose Framework Pipeline

This file describes how the main pipeline can be started and how the workflow of the pipeline is.

## Pipeline Execution
1. Run `ros2 launch stretch_funmap mapping.launch.py map_yaml:="/home/hello-robot/stretch_user/debug/merged_maps/merged_map_[timestamp]` with the map
2. Run `ros2 launch stretch_core d405_basic.launch.py` to be able to use the **gripper camera**
3. Start necessary dockers like **GPD/AnyGrasp** and **YOLO Drawer** in the Terminal of the workstation
4. Set *'Adaptable'* parameters at the top of `searchnet_execution.py` & **start script**
	

## Pipeline Workflow
todo