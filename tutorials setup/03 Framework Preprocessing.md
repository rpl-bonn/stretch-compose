# Stretch-Compose Framework Preprocessing

This file describes the necessary one-time preprocessing steps of creating the different point clouds for navigation & localization, and setting up the scene graph


## Point Cloud Setup
### 1. FUNMAP mapping
1. Place robot at **start position**
2. Run `ros2 launch stretch_funmap mapping.launch.py` 
3. Run `ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True`
4. While in the keyboard_teleop window, press the **Spacebar** to start a scan 
	1. A scan consists of: turn head, turn body, turn head
	2. The **3D map** in FUNMAP **updates twice** in each scan process
	3. **Move** the robot **around** & scan from different positions to increase map size & quality

### 2. Low-Res Robot Point Cloud
1. Place robot at the **same starting position**
2. Run `ros2 launch stretch_funmap mapping.launch.py map_yaml:="/home/hello-robot/stretch_user/debug/merged_maps/merged_map_[timestamp]` with the **previously generated map**
3. Run `ros2 run stretch_core keyboard_teleop --ros-args -p mapping_on:=True`
4. Run `ros2 run stretch_package pointcloud_service`
5. Run `ros2 run stretch_package pointcloud_client` on the **workstation** in a **VS Code Terminal** to receive one point cloud
	1.  **Move** the robot **around** with the keyboard_teleop & get point clouds from different views
	2. This will save single .ply files in the `data/autowalk_scans` folder
	3. *Note*: Fiducial needs to be visible in scans
6. Fill in the name of the folder in `autowalk_scans` in the **config** file under `pre_scanned_graphs/low_res`

### 3. High-Res IPad Point Cloud
1. Choose **LiDAR Advanced**, set Max Depth to ~2-3m and Resolution to ~6-10mm in the **3D Scanner App**
2. Scan environment and **process Scan HD** afterwards
	1. *Note*: Fiducial needs to be visible in scans
3. Save `Share->PointCloud->PLY` with `High Density` **selected** and `Z axis up` **disabled** into the scan folder as `mesh.ply`
4. Connect iPad to workstation & copy the whole scan folder into the `data/ipad_scans` folder
	1. *Note*: The structure should look like this: 
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
5. Fill in the name of the folder in `ipad_scans` in the **config** file under `pre_scanned_graphs/high_res`

### 4. Point Cloud Preprocessing
1. Start the **OpenMask3D** docker
2. Run `pointcloud_preprocessing.py` 
	1. This will **merge** the low-res point clouds, **align** it with the high-res point cloud & run the openmask3d **segmentation**


## Scene Graph Setup
1. Run the **Mask3D** docker once
2. Start the **YOLO Drawer** docker
3. Run `scenegraph_preprocessing.py`
	1. This will create the scene graph from the mask3d segmentation & save it to different **JSON files**

 ## [Optional] Visualize Centroid IDs
 Run `/bin/python /home/ws/source/scripts/preprocessing_scripts/centroid_position_finder.py`
