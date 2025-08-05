# Stretch-Compose Project Setup

This file describes how to setup the repository as a [Dev Container](https://docs.ros.org/en/humble/How-To-Guides/Setup-ROS-2-with-VSCode-and-Docker-Container.html#install-remote-development-extension) and which models and dockers are necessary to run the Stretch-Compose framework. It also describes how to configure the robot's URDF for [Inverse Kinematics](https://forum.hello-robot.com/t/inverse-kinematics-tutorial-workshop-recording/639).


## Dev Container Setup
1. **VS Code** and **Docker** need to be installed on the workstation
2. **Clone** the stretch-compose repository 
3. Open the project in VS Code & install the **Dev Containers** extension
4. Press `F1` to open the **command palette** & execute the command `Dev Containers: Rebuild without Cache and Reopen in Container` (This takes a few minutes to build)
5. Now the project **is running** inside the Dev Container
6. Run `export PYTHONPATH=$PYTHONPATH:/home/ws` in the VS Code Terminal to prevent the *ModuleNotFoundError: No module named 'source'*
7. *Note*: Maybe not all necessary Python libraries are already installed yet


## Model Setup
1. The **[Mask3D](https://github.com/behretj/Mask3D)** repository needs to be cloned into the `source` folder and setup like this: 
```
git clone https://github.com/behretj/Mask3D.git
mkdir Mask3D/checkpoints
cd Mask3D/checkpoints
wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
```
2. The **[SAM2](https://github.com/facebookresearch/sam2)** repository needs to be cloned into the `source` folder and setup like this:
```
git clone https://github.com/facebookresearch/sam2.git
cd sam2
pip install -e .
```
3. The **[GPD](https://github.com/rpl-bonn/gpd/tree/master)** repository needs to be cloned into the `source` folder. (Adapt the `run_docker_new.sh` and `cfg/ros_eigen_params.cfg` to fit your system!)


## Docker Setup
1. **Pull** the following docker images via `docker pull [link]` in the Terminal
2. **Run** the docker container & start the server via `docker run [command]` in the Terminal
| Name | Link | Command |
|------|------|---------|
| [Anygrasp](https://gitlab.uni-bonn.de/rpl/public_registry/container_registry/95) |`registry.gitlab.uni-bonn.de:5050/rpl/public_registry/graspnet:v1.0` | `--net=bridge --mac-address=02:42:ac:11:00:02 -p 5000:5000 --gpus all -it craiden/graspnet:v1.0 python3 app.py` |
| [GPD](https://gitlab.uni-bonn.de/rpl/public_registry/container_registry/143) |`registry.gitlab.uni-bonn.de:5050/rpl/public_registry/gpd:latest` | `cd` into the gpd folder & start with `run_docker_new.sh` instead of `docker run` |
| [OpenMask3D](https://hub.docker.com/layers/craiden/openmask/v1.0/images/sha256-023e04ebecbfeb62729352a577edc41a7f12dc4ce780bfa8b8e81eb54ffe77f7?context=repo) | `craiden/openmask:v1.0` | `-p 5001:5001 --gpus all -it craiden/openmask:v1.0 python3 app.py` |
| [Mask3D](https://hub.docker.com/layers/rupalsaxena/mask3d_docker/latest/images/sha256-db340d9db711334dc5b34c680bed59c0b4f068a95ce0476aa7edc1dd6ca906a3?context=repo) | `rupalsaxena/mask3d_docker:latest` | `--gpus all -it -v /home:/home -w /home/ws/source/Mask3D rupalsaxena/mask3d_docker:latest python3 mask3d.py --seed 42 --workspace /home/ws/data/prescans/<high_res_name>` |
| [YOLO Drawer](https://hub.docker.com/layers/craiden/yolodrawer/v1.0/images/sha256-2b0e99d77dab40eb6839571efec9789d6c0a25040fbb5c944a804697e73408fb?context=repo) |  `craiden/yolodrawer:v1.0` | `-p 5004:5004 --gpus all -it craiden/yolodrawer:v1.0 python3 app.py` |
| [ViTPose](https://hub.docker.com/layers/craiden/vitpose/v1.0/images/sha256-43a702300a0fffa2fb51fd3e0a6a8d703256ed2d507ac0ba6ec1563b7aee6ee7?context=repo) | `craiden/vitpose:v1.0` | `-p 5002:5002 --gpus all -it craiden/vitpose:v1.0 easy_ViTPose/venv/bin/python app.py` |


## URDF Setup 
1. Copy `stretch.urdf` file & `meshes` folder **from the Stretch robot** into the `data/stretch_description/original_urdf` folder
2. **Run** the `urdf_setup.py` file
3. Now the **new urdf** is saved into the `data/stretch_description/modified_urdf` folder
