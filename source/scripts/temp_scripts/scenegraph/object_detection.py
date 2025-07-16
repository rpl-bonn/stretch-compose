import numpy as np
import pickle, glob, os
from projectaria_tools.core import data_provider, calibration
from projectaria_tools.core.mps.utils import get_nearest_wrist_and_palm_pose, get_nearest_pose
import projectaria_tools.core.mps as mps

### all functions here are not really needed anymore


def get_all_aria_hand_poses(scan_dir, mode_left_right=False):
    """ Returns all Aria MPS poses for left and right hand """
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    wrist_and_palm_poses_path = scan_dir + "/mps_" + filename + "_vrs/hand_tracking/wrist_and_palm_poses.csv"
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

    hand_poses_left, hand_poses_right, T_poses = [], [], []
    both_count, i = 0, 0
    for hand_pose in wrist_and_palm_poses:
        query_timestamp = hand_pose.tracking_timestamp.total_seconds()*1e9
        device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)
        if device_pose:
            T_world_device = device_pose.transform_world_device.to_matrix()
            if hand_pose.left_hand.confidence > 0.0:
                left_palm_position_device = hand_pose.left_hand.palm_position_device
                left_palm_position_world = np.dot(T_world_device, np.append(left_palm_position_device, 1))[:3]
                hand_poses_left.append(left_palm_position_world)
                T_poses.append(T_world_device)
            if hand_pose.right_hand.confidence > 0.0:
                right_palm_position_device = hand_pose.right_hand.palm_position_device
                right_palm_position_world = np.dot(T_world_device, np.append(right_palm_position_device, 1))[:3]
                hand_poses_right.append(right_palm_position_world) if mode_left_right else hand_poses_left.append(right_palm_position_world)
                T_poses.append(T_world_device)
    
    return (np.array(hand_poses_left), np.array(hand_poses_right), T_poses) if mode_left_right else (np.array(hand_poses_left), T_poses)

def get_hand_object_interactions(scan_dir, mode_left_right=False):
    """ Returns nearest Aria MPS poses, separated for left and right hand, for all the interactions between
        hand and object. Hand-object interactions are filtered and in the end only the interactions are
        returned where the hand is in contact with the object, the object is movable, the hand and object
        are detected with high confidence and all poses lie within 100ms of each other """
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"

    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)

    detection_files = glob.glob(os.path.join(scan_dir, '*.pickle'))
    assert detection_files is not None, "No detection files found in directory"
    detection_path = detection_files[0]
    with open(detection_path, "rb") as f:
        detection_results = pickle.load(f)
    
    wrist_and_palm_poses_path = scan_dir + "/mps_" + filename + "_vrs/hand_tracking/wrist_and_palm_poses.csv"
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)

    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    hand_poses_left, hand_poses_right, T_world_devices = [], [], []
    
    for index in range(0, provider.get_num_data(stream_id)):
        name_curr = f"frame_{index:05}.jpg"
        image_info = detection_results[name_curr]
        hand_dets = image_info['hand_dets']
        obj_dets = image_info['obj_dets']

        if (obj_dets is not None) and (hand_dets is not None):
            correct_state = False
            for i in range(hand_dets.shape[0]):
                hand_bbox = list(int(np.round(x)) for x in hand_dets[i, :4])
                hand_score = hand_dets[i, 4]
                hand_state = hand_dets[i, 5]
                if hand_score > 0.8 and hand_state == 3: # hand is in contact
                    correct_state = True
                    break
                # hand_vec = hand_dets[i, 6:9]
                # hand_lr = hand_dets[i, -1]
            
            if not correct_state:
                continue

            found_object = False

            for i in range(obj_dets.shape[0]):
                # coordinates are a bit off (very strange)
                # obj_bbox = list(int(np.round(x)) for x in obj_dets[i, :4])
                obj_score = obj_dets[i, 4]
                if obj_score > 0.9:
                    found_object = True
                    break
            
            if not found_object:
                continue

            # get timestamp of current image
            query_timestamp = provider.get_image_data_by_index(stream_id, index)[1].capture_timestamp_ns
            
            device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)
            if device_pose is None:
                continue
            if abs(device_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) > 1e8:
                continue
            T_world_device = device_pose.transform_world_device.to_matrix()

            wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, query_timestamp)
            if wrist_and_palm_pose is None:
                continue

            found_object = False

            if wrist_and_palm_pose.left_hand.confidence > 0.5 and abs(wrist_and_palm_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) < 1e8:
                left_palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device
                left_palm_position_world = np.dot(T_world_device, np.append(left_palm_position_device, 1))[:3]
                hand_poses_left.append(left_palm_position_world)
                T_world_devices.append(T_world_device)
                found_object = True
            
            if wrist_and_palm_pose.right_hand.confidence > 0.5 and abs(wrist_and_palm_pose.tracking_timestamp.total_seconds()*1e9 - query_timestamp) < 1e8:
                right_palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device
                right_palm_position_world = np.dot(T_world_device, np.append(right_palm_position_device, 1))[:3]
                hand_poses_right.append(right_palm_position_world) if mode_left_right else hand_poses_left.append(right_palm_position_world)
                if not found_object:
                    T_world_devices.append(T_world_device)

    return (np.array(hand_poses_left), np.array(hand_poses_right), np.array(T_world_devices)) if mode_left_right else (np.array(hand_poses_left), np.array(T_world_devices))


def get_all_object_detections(scan_dir):
    """ Returns all object detections with corresponding hand poses """
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"

    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)
    calib = provider.get_device_calibration().get_camera_calib(camera_label)
    w, h = calib.get_image_size()

    pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

    detection_files = glob.glob(os.path.join(scan_dir, '*.pickle'))
    assert detection_files is not None, "No detection files found in directory"
    detection_path = detection_files[0]
    with open(detection_path, "rb") as f:
        detection_results = pickle.load(f)
    
    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)

    hands_left, objects, hands_right = [], [], []
    for i in range(0, provider.get_num_data(stream_id)):
        name_curr = f"frame_{i:05}.jpg"
        if detection_results[name_curr]['obj_dets'] is not None and detection_results[name_curr]['hand_dets'] is not None:
            query_timestamp = provider.get_image_data_by_index(stream_id, i)[1].capture_timestamp_ns
            img = provider.get_image_data_by_index(stream_id, i)[0].to_numpy_array()
            pose = get_nearest_pose(closed_loop_traj, query_timestamp)
            if pose is None:
                continue
            T_world_device = pose.transform_world_device
            T_device_camera = calib.get_transform_device_camera()
            T_world_camera = T_world_device @ T_device_camera
            T_world_camera = T_world_camera.to_matrix()
            obj = detection_results[name_curr]['obj_dets'][0]
            T_world_device = T_world_device.to_matrix()

            # pixel coordinates of object
            x, y = (obj[0] + obj[2])/2, (obj[1]+obj[3])/2
            # rotate pixel coordinate (camera model is for rotated image)
            obj_pos = y, w-x-1

            obj_pos = calib.unproject(obj_pos)
            # default value for z (depth)
            obj_pos[2] = 0.3
            objects.append(np.dot(T_world_device, np.append(obj_pos, 1))[:3])
            for detection in detection_results[name_curr]['hand_dets']:
                x, y = (detection[0]+detection[2])/2, (detection[1]+detection[3])/2
                obj_pos = y, w-x-1
                obj_pos = calib.unproject_no_checks(obj_pos)
                obj_pos[2] = 0.3
                if detection[-1] == 0:
                    hands_left.append(np.dot(T_world_device, np.append(obj_pos, 1))[:3])
                else:
                    hands_right.append(np.dot(T_world_device, np.append(obj_pos, 1))[:3])
    
    return hands_left, objects, hands_right


def get_first_detection(scan_dir):
    vrs_files = glob.glob(os.path.join(scan_dir, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    vrs_file = vrs_files[0]
    filename = os.path.splitext(os.path.basename(vrs_file))[0]

    provider = data_provider.create_vrs_data_provider(vrs_file)
    assert provider is not None, "Cannot open file"

    camera_label = "camera-rgb"
    stream_id = provider.get_stream_id_from_label(camera_label)

    detection_files = glob.glob(os.path.join(scan_dir, '*.pickle'))
    assert detection_files is not None, "No detection files found in directory"
    detection_path = detection_files[0]
    with open(detection_path, "rb") as f:
        detection_results = pickle.load(f)
    
    wrist_and_palm_poses_path = scan_dir + "/mps_" + filename + "_vrs/hand_tracking/wrist_and_palm_poses.csv"
    wrist_and_palm_poses = mps.hand_tracking.read_wrist_and_palm_poses(wrist_and_palm_poses_path)
    # valid_poses = [wrist_and_palm_poses[0].tracking_timestamp.total_seconds()*1e9, wrist_and_palm_poses[-1].tracking_timestamp.total_seconds()*1e9]

    closed_loop_path = scan_dir + "/mps_" + filename + "_vrs/slam/closed_loop_trajectory.csv"
    closed_loop_traj = mps.read_closed_loop_trajectory(closed_loop_path)
    
    for i in range(1, provider.get_num_data(stream_id)-1):
        name_prev = f"frame_{i-1:05}.jpg"
        name_curr = f"frame_{i:05}.jpg"
        name_next = f"frame_{i+1:05}.jpg"
        # very simple heuristic: if current, previous and next image have object detections -> found detection
        if detection_results[name_prev] and detection_results[name_curr] and detection_results[name_next]:
            if detection_results[name_prev]['obj_dets'] is not None and \
                detection_results[name_curr]['obj_dets'] is not None and \
                    detection_results[name_next]['obj_dets'] is not None:
                print(f"Found detection at frame {i}")

                # get timestamp of current image
                image_data = provider.get_image_data_by_index(stream_id, i)
                query_timestamp = image_data[1].capture_timestamp_ns

                # TODO: check whether hand an device pose are close enough to timestamp
                # get hand pose at this timestamp
                wrist_and_palm_pose = get_nearest_wrist_and_palm_pose(wrist_and_palm_poses, query_timestamp)
                if wrist_and_palm_pose is None:
                    print("No wrist and palm pose found")
                    continue

                device_pose = get_nearest_pose(closed_loop_traj, query_timestamp)
                if device_pose is None:
                    print("No device pose found")
                    continue
                
                T_world_device = device_pose.transform_world_device.to_matrix()

                # check whether left or right palm was detected with sufficient confidence
                left_palm_position_world = None
                if wrist_and_palm_pose.left_hand.confidence > 0.8:
                    left_palm_position_device = wrist_and_palm_pose.left_hand.palm_position_device
                    left_palm_position_world = np.dot(T_world_device, np.append(left_palm_position_device, 1))[:3]
                
                right_palm_position_world = None
                if wrist_and_palm_pose.right_hand.confidence > 0.8:
                    right_palm_position_device = wrist_and_palm_pose.right_hand.palm_position_device
                    right_palm_position_world = np.dot(T_world_device, np.append(right_palm_position_device, 1))[:3]
                
                if right_palm_position_world is None and left_palm_position_world is None:
                    print("No palm position found")
                    continue

                return left_palm_position_world, right_palm_position_world


    



    

