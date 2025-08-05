<h2 align="center"> Open-Vocabulary and Semantic-Aware Search and Retrieval of Objects using the Stretch Robot</h2>


# Project Structure

This project is setup as a vs code dev container. It has a ros2 workspace for ros communication with the stretch robot. The source folder contains the scripts for llm communication, preprocessing, and robot task execution as well as different utility functions and model interfaces.

```
stretch-compose/
├── configs/config.yaml 			# Config file
├── data/					# Different data
│   ├── aligned_point_clouds/			# Aligned high- & low-res point cloud
│   ├── autowalk_scans/             		# Single robot point clouds from autowalk
│   ├── images/                     		# Images taken with robot cameras
│   ├── ipad_scans/                 		# Raw prescan data from ipad
│   ├── merged_point_clouds/        		# Merged point clouds from autowalk
│   ├── openmask_features/          		# Mask3D output given aligned point clouds
│   ├── scene_graph/                		# Created scene graph from Mask3D
│   ├── stretch_description/        		# Robot urdf descriptions
├── ros2_ws/                                 	# ROS2 workspace
│   └── src/
│       ├── stretch_interface/srv/
│       │   ├── GetImage.srv
│       │   └── GetPointcloud.srv
│       └── stretch_package/stretch_package/	# Main ROS2 package for communicating with Stretch
│           ├── stretch_images/
│           ├── stretch_movement/
│           ├── stretch_pointclouds/
│           └── stretch_state/
└── source/                                 	# All source code
    ├── scripts/
    │   ├── llm_scripts/			# Large Language model scripts
    │   │   ├── deepseek_client.py
    │   │   ├── openai_client.py
    │   │   └── ...                         	# Other llm scripts
    │   ├── my_robot_scripts/			# Robot Pipeline Scripts
    │   │   ├── graspnet_planning.py        
    │   │   ├── graspnet_execution.py       
    │   │   ├── searchnet_planning.py 
    │   │   ├── searchnet_execution.py
    │   │   └── ...                         	# Other pipeline scripts 
    │   ├── preprocessing_scripts/		# Robot Preprocessing scripts
    │   │   ├── pointcloud_preprocessing.py
    │   │   └── scenegraph_preprocessing.py                      
    │   └── urdf_setup.py           		# URDF setup script          
    └── utils/                              	# General utility functions
        ├── preprocessing_utils/		# Preprocessing utility functions
        ├── robot_utils/  			# Robot utility functions                  
        │   ├── advanced_movement.py        	# Advanced movement robot commands
        │   ├── basic_movement.py           	# Basic movement robot commands
        │   ├── basic_perception.py         	# Basic perception robot commands
        │   ├── frame_transformer.py        	# Simplified transformation between frames of reference
        │   └── global_parameters.py        
        ├── camera_geometry.py              
        ├── coordinates.py                  	# Coordinate calculations (poses, translations, etc.)
        ├── docker_communication.py         
        ├── drawer_detection.py             
        ├── environment.py                 
        ├── files.py                        
        ├── graspnet_interface.py 		# Interface for graspnet          
        ├── importer.py                     
        ├── mask3D_interface.py             	# Interface for Mask3D
        ├── object_detection.py            
        ├── openmask_interface.py          	# Interface for OpenMask3D 
        ├── point_clouds.py                 	# Point cloud computations
        ├── recursive_config.py             	# Recursive configuration files
        ├── scannet_200_labels.py           
        ├── time.py                   
        ├── user_input.py                   
        ├── vis.py                          	# Handle visualizations
        ├── vitpose_interface.py         	# Interface for Vitpose   
        └── zero_shot_object_detection.py   	# Object detections from images                                                             
```

# Setup Instructions
All setup instructions for the robot, the workstation, and the framework can be found in the **tutorials setup** folder.
