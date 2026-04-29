"""
# 3D Reconstruction and Localization Demo
This notebook demonstrates:
1. Building a 3D model from images using triangulation
2. Visualizing the reconstruction
3. Localizing new images in the reconstructed model
"""

from pathlib import Path
import shutil
import json
import cv2
import numpy as np
import pycolmap

from hierarchical_localization import hloc

from hierarchical_localization.hloc import (
    extract_features,
    match_features,
    pairs_from_retrieval,
    triangulation,
    visualization,
    localize_sfm,
    reconstruction
)
from hierarchical_localization.hloc.visualization import plot_images, read_image
from hierarchical_localization.hloc.utils import viz_3d
from hierarchical_localization.hloc.localize_sfm import QueryLocalizer, pose_from_cluster
import h5py
import matplotlib.pyplot as plt
from utils.recursive_config import Config

# input data
config = Config()
scan_root = Path(config.get_subpath("ipad_scans"))
scan_date = config["pre_scanned_graphs"]["high_res"]
ipadscan = scan_root / scan_date

if not ipadscan.exists():
    raise FileNotFoundError(
        f"Scan directory not found: {ipadscan}. "
        "Check configs/config.yaml pre_scanned_graphs.high_res and data/ipad_scans."
    )

# get all images
jpg_files = sorted(ipadscan.glob("frame*.jpg"))
print(f"Found {len(jpg_files)} images in {ipadscan}")
if len(jpg_files) == 0:
    raise FileNotFoundError(
        f"No frame*.jpg files found in {ipadscan}. "
        "Add scan frames or update pre_scanned_graphs.high_res in configs/config.yaml."
    )

#setup proc directory with images
if (ipadscan / "proc").exists():
    shutil.rmtree(ipadscan / "proc")
images = ipadscan / "proc" / "images"
images.mkdir(parents=True, exist_ok=False)
for jpg in jpg_files:
    shutil.copy(jpg, images)


# Setup paths
#images = Path("/home/user/karimulla/hloc_repo/Hierarchical-Localization/matched_jpg_files_1")
#outputs = Path("/home/user/blumh/outputs_alisha")
outputs = ipadscan / "outputs"
outputs.mkdir(exist_ok=True, parents=True)

# Define all output paths
feature_path = outputs / "features.h5"
matches_path = outputs / "matches.h5"
#pairs_file = outputs / "pairs-retrieval.txt"
sfm_dir = outputs / "sfm"




def visualize_matches(matches_path, pairs_file, images_dir):
    """Visualize matches between image pairs"""
    with h5py.File(matches_path, 'r') as f:
        # Get first few pairs
        with open(pairs_file, 'r') as pf:
            pairs = [line.strip().split() for line in pf][:5]

        for pair in pairs:
            img1_name, img2_name = pair
            img1 = read_image(images_dir / img1_name)
            img2 = read_image(images_dir / img2_name)

            pair_name = f'{img1_name}_{img2_name}'
            if pair_name in f:
                matches = f[pair_name][:]
                print(f"Number of matches for {pair_name}: {len(matches)}")

                # Plot images side by side with matches
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
                ax1.imshow(img1)
                ax2.imshow(img2)
                ax1.set_title(img1_name)
                ax2.set_title(img2_name)
                plt.show()

"""
## 1. Feature Extraction and Matching
"""
# Extract local features
print("\nExtracting features...")
# feature_conf = extract_features.confs["superpoint_aachen"]
feature_conf = {
    'output': 'feats-superpoint',
    'model': {
        'name': 'superpoint',
        'nms_radius': 3,
        'max_keypoints': 8192,  # Increased from default
        'keypoint_threshold': 0.005  # Lower threshold to get more keypoints
    },
    'preprocessing': {
        'grayscale': True,
        'resize_max': 1600,
    }
}
features = extract_features.main(
    conf=feature_conf,
    image_dir=images,
    export_dir=outputs,
    feature_path=feature_path
)

# Extract global descriptors for retrieval
global_conf = extract_features.confs["netvlad"]
global_path = outputs / "global-descriptors.h5"
global_descriptors = extract_features.main(
    conf=global_conf,
    image_dir=images,
    export_dir=outputs,
    feature_path=global_path
)

# Generate pairs using retrieval
print("\nGenerating image pairs...")
# First generate initial pairs
pairs_file = outputs / "pairs-retrieval.txt"
pairs_from_retrieval.main(
    descriptors=global_path,
    output=pairs_file,
    num_matched=10
)

# Read existing pairs and convert to set of tuples
#with open(pairs_file, 'r') as f:
    #existing_pairs = set()
    #for line in f:
        #if line.strip():  # Skip empty lines
            #parts = line.strip().split()
            #if len(parts) >= 2:  # Ensure line has at least 2 parts
                #existing_pairs.add((parts[0], parts[1]))

# Add sequential pairs
#image_list = sorted([p.name for p in images.glob('*.jpg')])
#sequential_pairs = set()
#for i in range(len(image_list)-1):
    #sequential_pairs.add((image_list[i], image_list[i+1]))

# Combine all pairs and write to file
#all_pairs = existing_pairs.union(sequential_pairs)

# Write pairs in correct format (one pair per line)
#with open(pairs_file, 'w') as f:
    #for img1, img2 in sorted(all_pairs):
        #f.write(f'{img1} {img2}\n')

#print(f"Total number of pairs: {len(all_pairs)}")

# Match features
print("\nMatching features...")
# matcher_conf = match_features.confs["superglue"]
matcher_conf = {
    'output': 'matches-superglue',
    'model': {
        'name': 'superglue',
        'weights': 'indoor', # 'indoor' weights for better performance on indoor scenes
        'sinkhorn_iterations': 50,
        'match_threshold': 0.2,  # Lower threshold to get more matches
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9  # Increase number of layers
    },
    'num_workers': 0,
    'writer_threads': 1,
    'pin_memory': False,
}
matches = match_features.main(
    conf=matcher_conf,
    pairs=pairs_file,
    features=feature_path,
    export_dir=outputs,
    matches=matches_path
)

# For sequential pairs, only match consecutive frames with small overlap
#sequence_length = 3  # Match with 3 frames before and after
#for i in range(len(image_list)):
    #for j in range(max(0, i-sequence_length), min(i+sequence_length+1, len(image_list))):
        #if i != j:
            #sequential_pairs.add((image_list[i], image_list[j]))

#visualize_matches(matches_path, pairs_file, images)
"""
## 2. Create Reference Model from Camera Parameters
"""
def create_reference_model(ipad_dir, image_dir):



    reconstruction = pycolmap.Reconstruction()

    for imagefile in sorted(image_dir.glob('*.jpg')):
        image_name = imagefile.name.split('.jpg')[0]
        json_file = ipadscan / f'{image_name}.json'
        if not json_file.exists():
            print(imagefile)
            assert False
        with open(json_file, 'r') as f:
            data = json.load(f)
        # OpenCV returns image shape as (height, width, channels).
        height, width = cv2.imread(str(imagefile)).shape[:2]
        fx = float(data['intrinsics'][0])
        fy = float(data['intrinsics'][4])
        cx = float(data['intrinsics'][2])
        cy = float(data['intrinsics'][5])

        # Create camera
        camera = pycolmap.Camera.create(
            camera_id=len(reconstruction.cameras)+1,
            model=pycolmap.CameraModelId.PINHOLE,
            focal_length=fx,
            width=width,
            height=height
        )
        camera.params = [fx, fy, cx, cy]
        reconstruction.add_camera_with_trivial_rig(camera)

        image = pycolmap.Image(
            name=imagefile.name,
            camera_id=camera.camera_id,
            image_id=len(reconstruction.images)+1,
        )

        # Set pose
        pose = np.asarray(data['cameraPoseARFrame']).reshape((4, 4))

        rotation = pose[:3, :3]

        # rotate the camera and flip the axis
        rotation[2, :] = -rotation[2, :]
        rotation[1, :] = -rotation[1, :]
        rotation[0, :] = -rotation[0, :]
        pose[:3, :3] = rotation
        pose[:, 0] = -pose[:, 0]

        R_x_90 = np.array([[1, 0, 0, 0],
            [0, np.cos(np.radians(90)), -np.sin(np.radians(90)), 0],
            [0, np.sin(np.radians(90)), np.cos(np.radians(90)), 0],
            [0, 0, 0, 1]])

        R_z_90 = np.array([[np.cos(np.radians(-90)), -np.sin(np.radians(-90)), 0, 0],
            [np.sin(np.radians(-90)), np.cos(np.radians(-90)), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]])

        R_z_x = np.dot(R_z_90, R_x_90)
        pose = np.dot(R_z_x, pose)
        pose = np.linalg.inv(pose)
        cam_from_world = pycolmap.Rigid3d(pose[:3])
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction

# Create reference model
print("\nCreating reference model...")
reference_model = create_reference_model(ipadscan, images)
ref_model_path = ipadscan / "proc" / "reference_model"
ref_model_path.mkdir(exist_ok=True)
reference_model.write_text(str(ref_model_path))

"""
## 3. Run Triangulation
"""
print("\nRunning triangulation...")

# model = reconstruction.main(
#     sfm_dir, images, pairs_file, feature_path, matches_path,
# )
# out_model_path = ipadscan / "proc" / "sfm_model"
# out_model_path.mkdir(exist_ok=True)
# model.write_text(str(out_model_path))
# model = pycolmap.Reconstruction(str(ipadscan / "outputs" / "sfm"))

model = triangulation.main(
    sfm_dir=sfm_dir,
    # reference_model=(ipadscan / "outputs" / "sfm"),
    reference_model=ref_model_path,
    image_dir=images,
    pairs=pairs_file,
    features=feature_path,
    matches=matches_path,
    skip_geometric_verification=True
)

"""
## 4. Visualize Reconstruction
"""
if model is not None:
    print("\nVisualization: 3D Model")

    # Create 3D visualization
    fig = viz_3d.init_figure()
    viz_3d.plot_reconstruction(
        fig,
        model,
        color="rgba(255,0,0,0.5)",
        name="mapping",
        points=len(model.points3D) > 0,
        cameras=True,
        cs=0.1,
        points_rgb=True
    )

    fig.update_layout(
        scene=dict(
            aspectmode='data',
            camera=dict(
                up=dict(x=0, y=1, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            ),
        ),
        title="3D Reconstruction"
    )
    fig.show()

    # Visualize feature matches
    print("\nVisualization: Feature Matches")
    visualization.visualize_sfm_2d(
        model,
        images,
        color_by="visibility",
        n=2
    )

    print("\nReconstruction Statistics:")
    print(model.summary())

    # Save reconstruction
    output_path = sfm_dir / "reconstruction"
    output_path.mkdir(exist_ok=True, parents=True)
    model.write(str(output_path))
    print(f"\nReconstruction saved to: {output_path}")

"""
## 5. Localize a Query Image
"""
def localize_image(query_path, reconstruction, feature_path, query_root=None):
    query_feature_path = outputs / "query_features.h5"
    query_matches_path = outputs / "query_matches.h5"

    # Extract features for query into a separate file
    extract_features.main(
        feature_conf,
        query_root or images,
        image_list=[query_path.name],
        feature_path=query_feature_path,
        overwrite=True
    )

    # Create pairs for query
    query_pairs = outputs / "query_pairs.txt"
    with open(query_pairs, 'w') as f:
        for _, ref_image in reconstruction.images.items():
            f.write(f'{query_path.name} {ref_image.name}\n')

    # Match query features against reference features into a separate file
    match_features.main(
        matcher_conf,
        query_pairs,
        features=query_feature_path,
        features_ref=feature_path,
        matches=query_matches_path,
        overwrite=True
    )

    # Setup localizer
    camera = pycolmap.infer_camera_from_image(query_path)
    ref_ids = list(reconstruction.images.keys())
    conf = {
        "estimation": {"ransac": {"max_error": 12}},
        "refinement": {"refine_focal_length": True, "refine_extra_params": True},
    }

    # Localize
    localizer = QueryLocalizer(reconstruction, conf)
    ret, log = pose_from_cluster(
        localizer,
        query_path.name,
        camera,
        ref_ids,
        query_feature_path,
        query_matches_path
    )

    return ret, log, camera

# Example of localizing a new image (replace with your query image)
# query_root = images  # Replace with your query image
# query_path = query_root / "frame_00065.jpg" 
query_root = Path("/home/ws/data/images/test_robot_images")
query_path = query_root / "test_robot_headcam_frame.png"
print(f"\nLocalizing query image: {query_path}")
if query_path.exists():
    print("\nLocalizing query image...")
    ret, log, camera = localize_image(query_path, model, feature_path, query_root=query_root)

    if ret is not None:
        total_corr = len(log.get("points3D_ids", []))
        print(f'Found {ret["num_inliers"]}/{total_corr} inlier correspondences.')

        # Visualize localization results
        visualization.visualize_loc_from_log(query_root, query_path.name, log, model, db_image_dir=images)

        # Add query camera to 3D visualization
        viz_3d.plot_camera_colmap(
            fig, ret["cam_from_world"], camera,
            color="rgba(0,255,0,0.5)",
            name="query",
            fill=True,
        )
        # Add inlier 3D points
        if "points3D_ids" in log:
            inlier_sel = ret.get("inliers", ret.get("inlier_mask"))
            if inlier_sel is not None:
                inlier_sel = np.asarray(inlier_sel)
                if inlier_sel.dtype == np.bool_:
                    inlier_sel = np.where(inlier_sel)[0]
                point3d_ids = np.asarray(log["points3D_ids"])
                inlier_sel = inlier_sel[inlier_sel < len(point3d_ids)]
                inl_3d = np.array([
                    model.points3D[pid].xyz
                    for pid in point3d_ids[inlier_sel]
                    if pid in model.points3D
                ])
                if len(inl_3d):
                    viz_3d.plot_points(fig, inl_3d, color="lime", ps=1, name="query_points")

        fig.show()
        
else:
    print(f"Query image not found: {query_path}. Please check the path and try again.")