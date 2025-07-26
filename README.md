<h2 align="center"> Open-Vocabulary and Semantic-Aware Search and Retrieval of Objects with the Stretch Robot</h2>


# Code Structure

This project has a ros2 workspace for ros communication with the Stretch robot. The source folder contains the scripts to be run for setup, point cloud preprocessing, and robot task execution as well as different utility functions.

```
stretch-compose/
├── ros2_ws/                                # ROS2 workspace
│   └── src/
│       ├── stretch_interface/srv/
│       │   ├── GetImage.srv
│       │   └── GetPointcloud.srv
│       └── stretch_package/stretch_package/
│           ├── stretch_images/
│           ├── stretch_movement/
│           ├── stretch_pointclouds/
│           └── stretch_state/
├── source/                                 # All source code
│   ├── scripts/
│   │   ├── llm_scripts/
│   │   │   ├── deepseek_client.py
│   │   │   ├── openai_client.py
│   │   │   └── ...                         # Other llm scripts
│   │   ├── my_robot_scripts/
│   │   │   ├── graspnet_planning.py        
│   │   │   ├── graspnet_execution.py       
│   │   │   ├── searchnet_planning.py 
│   │   │   ├── searchnet_execution.py
│   │   │   └── ...                         # Other pipeline scripts 
│   │   ├── preprocessing_scripts/
│   │   │   ├── pointcloud_preprocessing.py
│   │   │   └── scenegraph_preprocessing.py     
│   │   ├── mnist_setup.py                  
│   │   └── urdf_setup.py                     
│   └── utils/                              # General utility functions
│       ├── preprocessing_utils/
│       ├── robot_utils/                    
│       │   ├── advanced_movement.py        # Advanced movement robot commands
│       │   ├── basic_movement.py           # Basic movement robot commands
│       │   ├── basic_perception.py         # Basic perception robot commands
│       │   ├── frame_transformer.py        # Simplified transformation between frames of reference
│       │   └── global_parameters.py        
│       ├── camera_geometry.py              
│       ├── coordinates.py                  # Coordinate calculations (poses, translations, etc.)
│       ├── docker_communication.py         
│       ├── drawer_detection.py             
│       ├── environment.py                 
│       ├── files.py                        
│       ├── graspnet_interface.py           
│       ├── importer.py                     
│       ├── mask3D_interface.py             
│       ├── object_detection.py            
│       ├── openmask_interface.py           
│       ├── point_clouds.py                 # Point cloud computations
│       ├── recursive_config.py             # Recursive configuration files
│       ├── scannet_200_labels.py           
│       ├── time.py                   
│       ├── user_input.py                   
│       ├── vis.py                          # Handle visualizations
│       ├── vitpose_interface.py            
│       └── zero_shot_object_detection.py   # Object detections from images              
├── configs/                                                        
└── shells/                                                          
```

# Setup Instructions
The project is build upon a [ROS2 Dev Container](https://docs.ros.org/en/humble/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html) with VSCode and Docker using `Python 3.10`.


## Model Setups
The pre-trained model weights for Yolov-based drawer detection are available [here](https://drive.google.com/file/d/11axGmSgb3zmUtq541hH2TCZ54DwTEiWi/view?usp=drive_link).

The [Mask3D](https://github.com/behretj/Mask3D) repository can be cloned into the `source` folder and setup like this:
```bash
git clone https://github.com/behretj/Mask3D.git
mkdir Mask3D/checkpoints
cd Mask3D/checkpoints
wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
```

The [SAM2](https://github.com/facebookresearch/sam2) repository also needs to be cloned into the `source` folder and setup like this:
```bash
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```

For grasping with [GPD](https://github.com/rpl-bonn/gpd/tree/master) pull and run the docker image as described in the repository. Also clone the repository into the `source` folder and adapt the `run_docker_new.sh` and `cfg/ros_eigen_params.cfg` to fit for your system. To use the docker, start the `run_docker_new.sh` in the terminal.

## Docker Setup
Docker containers are used to run external models. This allows for easy modularity when working with multiple methods, without tedious setup.
Each docker container functions as a self-contained server, answering requests. Please refer to `utils/docker_communication.py` for your own custom setup, 
or to the respective files in `utils/` for existing containers.

To run the respective docker container, please first pull the desired image via 
```bash
docker pull [Link]
```
Once docker has finished pulling the image, you can start a container via the `Run Command` in the terminal.
When you are inside the container shell, simply run the `Start Command` to start the server.

|      Name       |                                                                                   Link                                                                                   |                             Run Command                              |               Start Command               |
|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|:-----------------------------------------:|
|    AnyGrasp     | [craiden/graspnet:v1.0](https://hub.docker.com/layers/craiden/graspnet/v1.0/images/sha256-ec5663ce991415a51c34c00f2ea6f8ab9303a88e6ac27d418df2193c6ab40707?context=repo) |  ```docker run --net=bridge --mac-address=02:42:ac:11:00:02 -p 5000:5000 --gpus all -it craiden/graspnet:v1.0```  |           ```python3 app.py```            |
|   OpenMask3D    | [craiden/openmask:v1.0](https://hub.docker.com/layers/craiden/openmask/v1.0/images/sha256-023e04ebecbfeb62729352a577edc41a7f12dc4ce780bfa8b8e81eb54ffe77f7?context=repo) |  ```docker run -p 5001:5001 --gpus all -it craiden/openmask:v1.0```  |           ```python3 app.py```            |
|     ViTPose     |  [craiden/vitpose:v1.0](https://hub.docker.com/layers/craiden/vitpose/v1.0/images/sha256-43a702300a0fffa2fb51fd3e0a6a8d703256ed2d507ac0ba6ec1563b7aee6ee7?context=repo)  |  ```docker run -p 5002:5002 --gpus all -it craiden/vitpose:v1.0```   | ```easy_ViTPose/venv/bin/python app.py``` |
| DrawerDetection | [craiden/yolodrawer:v1.0](https://hub.docker.com/layers/craiden/yolodrawer/v1.0/images/sha256-2b0e99d77dab40eb6839571efec9789d6c0a25040fbb5c944a804697e73408fb?context=repo) | ```docker run -p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0``` |           ```python3 app.py```            |
| Mask3D | [rupalsaxena/mask3d_docker:latest](https://hub.docker.com/layers/rupalsaxena/mask3d_docker/latest/images/sha256-db340d9db711334dc5b34c680bed59c0b4f068a95ce0476aa7edc1dd6ca906a3?context=repo) | ```docker run --gpus all -it -v /home:/home -w /home/ws/source/Mask3D rupalsaxena/mask3d_docker:latest``` |           ```python3 mask3d.py --seed 42 --workspace /home/ws/data/prescans/<high_res_name>```            |


## Stretch Robot Setup
First time: Before using the robot for the first time, you need to connect to it to your workstation via the untethered setup, described [here](https://docs.hello-robot.com/0.3/getting_started/connecting_to_stretch/). 

Every time: After turning on the robot, run the following commands in the terminal before launching anything else:
```bash
stretch_free_robot_process.py
stretch_robot_home.py
stretch_system_check.py
```

## Network Setup
In this project setup, robot and workstation are connected via a separate router. The robot is connected via Wi-Fi and the workstation via cable. They communicate via `ROS2 humble`, so the `ROS_DOMAIN_ID` on the robot needs to be the same as in the `devcontainer.json`.


## URDF Setup
To use inverse kinematics, the urdf of the robot has to be modified. First, copy the urdf from the Stretch robot (`stretch.urdf` and `meshes` folder) into the `data/stretch_description/original_urdf` folder. Run the `urdf_setup.py` file in the `source/scripts` folder. This will save the new urdf into the `data/stretch_description/modified_urdf` folder.


## Point Cloud Setup
For this project, two point clouds are required for localization/navigation (low resolution, captured by Stretch) and segmentation (high resolution, captured by iPad). Before running any scripts, the point clouds have to be captured and preprocessed in the following way:


### FUNMAP mapping
First, a [FUNMAP](https://docs.hello-robot.com/0.2/stretch-ros/stretch_funmap/) map needs to be created on the Stretch robot for localization and navigation. 
1. Place the robot at its starting location.
2. Launch `stretch_funmap mapping` and `stretch_core keyboard_teleop` on the robot. 
3. Drive the robot around in the environment and initiate scans with the spacebar.

### Low-Resolution Stretch Point Cloud
As FUNMAP only results in Max Height Images and no Point Cloud, we need to do another scan with the robot. Make sure the fiducial is visible during the scan.
1. Place the robot at its starting location.
2. Launch `stretch_funmap mapping` with the previously generated map and `stretch_core keyboard_teleop` on the robot.
3. Run `pointcloud_service` on the robot to get point clouds.
4. Drive the robot around in the environment and request point clouds by running `pointcloud_client` on the workstation.
5. Fill in the name of the folder in `autowalk_scans` in the config file under `pre_scanned_graphs/low_res`.

### High-Resolution iPad Point Cloud
To capture the point cloud we use the [3D Scanner App](https://apps.apple.com/us/app/3d-scanner-app/id1419913995) on iOS. Make sure the fiducial is visible during the scan.
1. Choose LiDAR Advanced, set Max Depth to ~2.5-3m and Resolution to ~6-10mm in the 3D Scanner App.
2. Scan environment and process Scan HD afterwards.
3. Save `Share->PointCloud->PLY` with `High Density` enabled and `Z axis up` disabled into the scan folder as `mesh.ply`.
4. Copy LiDAR scan folder into the `data/ipad_scans` folder. The folder structure should look like this:
```
ipad_scans/
├── folder/
│   ├── annotations.json
│   ├── export.obj
│   ├── export_refined.obj
│   ├── frame_00000.jpg
│   ├── frame_00000.json
│   ├── ...
│   ├── info.json
│   ├── mesh.ply
│   ├── textured_output.jpg
│   ├── textured_output.mtl
│   ├── textured_output.obj
│   ├── thumb_00000.jpg                
│   └── world_map.arkit
```
5. Fill in the name of the folder in `ipad_scans` in the config file under `pre_scanned_graphs/high_res`.

### Point Cloud Preprocessing
To merge the Stretch scans into one point cloud and align it with the iPad point cloud, you need to run `pointcloud_preprocessing.py` in the `source/scripts/preprocessing_scripts` folder. This will also run the OpenMask3D Segmentation. 

After this, your data should contain the following folders:
```
data/
├── aligned_point_clouds/
├── autowalk_scans/             # Raw autowalk data
├── images/                     # Images taken with robot cameras
├── ipad_scans/                 # Raw prescan data from ipad
├── merged_point_clouds/        # Merged point clouds from autowalk
├── openmask_features/          # Mask3D output given aligned point clouds
├── scene_graph/                # Created scene graph from Mask3D
└── stretch_description/        # Robot urdf descriptions
```

### Scene Graph Creation
To create the scene graph, first run the Mask3D container once. Afterwards, start the YOLO Drawer docker and run the `scenegraph_preprocessing.py` in the `source/scripts/preprocessing_scripts` folder. This will also save the scene graph to different JSON files.

## Pipeline Execution
To run a robot task, launch `stretch_funmap mapping` with the generated map and `stretch_core d405_basic` on the robot. Then, start your desired task execution script from the `source/scripts/my_robot_scripts` folder on the workstation.