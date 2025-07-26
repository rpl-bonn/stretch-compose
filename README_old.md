<div align='center'>
<h2 align="center"> Open-Vocabulary and Semantic-Aware Search and Retrieval of Objects with the Stretch Robot</h2>
Stretch-Compose presents a comprehensive framework for integration of modern machine perception techniques with Stretch, 
showing experiments with object grasping, object search, and dynamic drawer and door manipulation.
</div>

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
│   │   ├── point_cloud_scripts/
│   │   │   ├── full_align.py
│   │   │   ├── full_merge.py
│   │   │   └── point_cloud_preprocessing.py
│   │   ├── robot_scripts/
│   │   │   ├── graspnet_planning.py        # ...
│   │   │   ├── graspnet_execution.py       # ...
│   │   │   └── ...                         # Other action scripts
│   │   ├── mnist_setup.py                  # ...
│   │   └── urdf_setup.py                   # ...   
│   └── utils/                              # General utility functions
│       ├── robot_utils/                    # Utility functions specific to stretch functionality
│       │   ├── advanced_movements.py       # Advanced robot commands (planning, complex movements)
│       │   ├── base.py                     # Framework and wrapper for all scripts
│       │   ├── basic_movements.py          # Basic robot commands (moving body / arm, stowing, etc.)
│       │   ├── frame_transformer.py        # Simplified transformation between frames of reference
│       │   ├── global_parameters.py        # ...
│       │   └── video.py                    # Handle actions that require access to robot cameras
│       ├── camera_geometry.py              # ...
│       ├── coordinates.py                  # Coordinate calculations (poses, translations, etc.)
│       ├── docker_communication.py         # Communication with docker servers
│       ├── drawer_detection.py             # ...
│       ├── environment.py                  # API keys, env variables
│       ├── files.py                        # File system handling
│       ├── graspnet_interface.py           # Communication with graspnet server
│       ├── importer.py                     # Config-based importing
│       ├── logger.py                       # ...
│       ├── logs.py                         # ...
│       ├── mask3D_interface.py             # Handling of Mask3D instance segmentation
│       ├── object_detection.py             # ...
│       ├── openmask_interface.py           # ...
│       ├── point_clouds.py                 # Point cloud computations
│       ├── recursive_config.py             # Recursive configuration files
│       ├── scannet_200_labels.py           # Scannet200 labels (for Mask3D)
│       ├── singletons.py                   # Singletons for global unique access
│       ├── user_input.py                   # Handle user input
│       ├── vis.py                          # Handle visualizations
│       ├── vitpose_interface.py            # Handle communications with VitPose docker server
│       └── zero_shot_object_detection.py   # Object detections from images              
├── configs/                                # Configs
│   └── config.yaml                         # Base configurations
├── README.md                               # Project documentation
├── requirements.txt                        # pip requirements file
├── pyproject.toml                          # Formatter and linter specs
└── LICENSE
```

# Setup Instructions
The project is build upon a [ROS2 Dev Container](https://docs.ros.org/en/iron/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html) with VSCode and Docker.

The main dependencies of the project are the following:
```yaml
python: 3.10
```
The pre-trained model weights for Yolov-based drawer detection are available [here](https://drive.google.com/file/d/11axGmSgb3zmUtq541hH2TCZ54DwTEiWi/view?usp=drive_link).


## Docker Setup
Docker containers are used to run external neural networks. This allows for easy modularity when working with multiple methods, without tedious setup.
Each docker container functions as a self-contained server, answering requests. Please refer to `utils/docker_communication.py` for your own custom setup, 
or to the respective files in `utils/` for existing containers.

To run the respective docker container, please first pull the desired image via 
```bash
docker pull [Link]
```
Once docker has finished pulling the image, you can start a container via the `Run Command`.
When you are inside the container shell, simply run the `Start Command` to start the server.

|      Name       |                                                                                   Link                                                                                   |                             Run Command                              |               Start Command               |
|:---------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|:--------------------------------------------------------------------:|:-----------------------------------------:|
|    AnyGrasp     | [craiden/graspnet:v1.0](https://hub.docker.com/layers/craiden/graspnet/v1.0/images/sha256-ec5663ce991415a51c34c00f2ea6f8ab9303a88e6ac27d418df2193c6ab40707?context=repo) |  ```docker run -p 5000:5000 --gpus all -it craiden/graspnet:v1.0```  |           ```python3 app.py```            |
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
To use inverse kinematics, the urdf of the robot has to be modified. First, copy the urdf from the Stretch robot (`stretch.urdf` and `meshes` folder) into the `data/stretch_description/urdf` folder. Run the `urdf_setup.py` file in the `source/scripts` folder. This will save the new urdf into the `data/stretch_description/tmp` folder.


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
1. Choose LiDAR Advanced, set Max Depth to ~2.5m and Resolution to ~6mm in the 3D Scanner App.
2. Scan environment and process Scan HD afterwards.
3. Save `Share->PointCloud->PLY` with `High Density` enabled and `Z axis up` disabled into the scan folder as `pcd.ply`.
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
│   ├── pcd.ply
│   ├── textured_output.jpg
│   ├── textured_output.mtl
│   ├── textured_output.obj
│   ├── thumb_00000.jpg                
│   └── world_map.arkit
```
5. Fill in the name of the folder in `ipad_scans` in the config file under `pre_scanned_graphs/high_res`.

### Point Cloud Preprocessing
To merge the Stretch scans into one point cloud and align it with the iPad point cloud, you need to run `pointcloud_preprocessing.py` in the `source/scripts/point_cloud_scripts` folder. This will also run the OpenMask3D Segmentation. 

After this, your data should contain the following folders:
```
├── data/
│   ├── aligned_point_clouds/               # IPad point clouds aligned with merged autowalk clouds
│   ├── autowalk_scans/                     # Raw autowalk data from robot
│   ├── images/                             # ...
│   ├── ipad_scans/                         # Raw prescan data from ipad
│   ├── merged_point_clouds/                # Extracted point clouds from autowalk
│   ├── openmask_features/                  # Mask3D output given aligned point clouds
│   └── stretch_description/                # Robot urdf
```

## Mask3D
Clone and setup the Mask3D repository like this:
```bash
cd $SPOTLIGHT/source/
git clone https://github.com/behretj/Mask3D.git
mkdir Mask3D/checkpoints
cd Mask3D/checkpoints
wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
```
Copy the following files into the `data/ipad_scans/<high_res_name>` folder:
```
mask3d_label_mapping.csv
data/aligned_point_clouds/<high_res_name>/pose/icp_tform_ground.txt
data/prescans/<high_res_name>/pcd.ply (and rename into mesh.ply)
```
Run the Mask3D docker.

