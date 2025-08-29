# Stretch-Compose Framework Pipeline

This file describes how the main pipeline can be started and how the workflow of the pipeline is.

## Pipeline Execution
1. Run `ros2 launch stretch_funmap mapping.launch.py map_yaml:="/home/hello-robot/stretch_user/debug/merged_maps/merged_map_[timestamp]` with the map
2. Run `ros2 launch stretch_core d405_basic.launch.py` to be able to use the **gripper camera**
3. Start necessary dockers like **GPD/AnyGrasp** and **YOLO Drawer** in the Terminal of the workstation
4. Set *'Adaptable'* parameters at the top of `searchnet_execution.py` & **start script**
	

## Pipeline Workflow
1. The robot looks for the searched object in the scene graph
2. If not found, it asks Deepseek for more possible locations 
3. Then it drives in front of all the determined furniture to check whether the object is in any of the open spaces
4. If yes, the robot moves closer & creates a point cloud of the object from different view poses
5. The robot generates a grasp pose & tries to grasp the object
6. If no, the robot looks for drawers in the already checked furniture
7. It drives in front of every drawer, opens it & looks into it, until it has found the object
