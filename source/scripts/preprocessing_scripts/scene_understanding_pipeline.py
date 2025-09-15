import numpy as np
import torch
import cv2
import json
from pathlib import Path
from ultralytics import YOLOWorld, SAM
import clip
from sklearn.cluster import KMeans
import open3d as o3d
from scipy.spatial import KDTree

class SceneUnderstandingPipeline:
    def __init__(self, data_folder, object_categories):
        self.data_folder = Path(data_folder)
        self.object_categories = object_categories
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load models
        self.yolo = YOLOWorld('yolo_world_l.pt')
        self.sam = SAM('sam2_l.pt')
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Data structures
        self.objects_3d = []  # Stores 3D object information
        self.scene_graph = {}
        
    def load_frame_data(self, frame_id):
        """Load RGB, depth, pose, and intrinsics for a frame"""
        rgb = cv2.imread(str(self.data_folder / 'rgb' / f'frame_{frame_id:06d}.png'))
        depth = np.load(str(self.data_folder / 'depth' / f'frame_{frame_id:06d}.npy'))
        pose = np.loadtxt(str(self.data_folder / 'pose' / f'frame_{frame_id:06d}.txt'))
        intrinsics = np.loadtxt(str(self.data_folder / 'intrinsics.txt'))
        
        return rgb, depth, pose, intrinsics
    
    def detect_objects_2d(self, rgb_image):
        """Detect objects and generate masks using YOLO-World + SAM2"""
        # Set custom categories for YOLO-World
        self.yolo.set_classes(self.object_categories)
        
        # Run detection
        results = self.yolo(rgb_image)
        detections = results[0]
        
        object_masks = []
        object_info = []
        
        for box in detections.boxes:
            # Extract bounding box
            bbox = box.xyxy[0].cpu().numpy()
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            
            if confidence < 0.3:  # Confidence threshold
                continue
            
            # Generate mask with SAM2
            sam_result = self.sam(rgb_image, bboxes=bbox[None, :])
            mask = sam_result[0].masks.data[0].cpu().numpy()
            
            object_masks.append(mask)
            object_info.append({
                'bbox': bbox,
                'confidence': confidence,
                'class_id': class_id,
                'category': self.object_categories[class_id]
            })
        
        return object_masks, object_info
    
    def extract_clip_features(self, rgb_image, mask):
        """Extract CLIP features for masked region"""
        # Apply mask to image
        masked_image = rgb_image.copy()
        masked_image[~mask.astype(bool)] = 0
        
        # Preprocess for CLIP
        image_rgb = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
        image_preprocessed = self.clip_preprocess(Image.fromarray(image_rgb)).unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_preprocessed)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def lift_to_3d(self, mask, depth, pose, intrinsics):
        """Convert 2D mask to 3D points using depth information"""
        # Get pixel coordinates from mask
        y_coords, x_coords = np.where(mask > 0)
        
        if len(y_coords) == 0:
            return None
        
        # Get depth values for masked pixels
        depth_values = depth[y_coords, x_coords]
        valid_depth = depth_values > 0
        
        if not np.any(valid_depth):
            return None
        
        # Convert to 3D camera coordinates
        x_coords = x_coords[valid_depth]
        y_coords = y_coords[valid_depth]
        depth_values = depth_values[valid_depth]
        
        # Camera coordinates
        points_camera = np.column_stack([
            (x_coords - intrinsics[0, 2]) * depth_values / intrinsics[0, 0],
            (y_coords - intrinsics[1, 2]) * depth_values / intrinsics[1, 1],
            depth_values
        ])
        
        # Transform to world coordinates
        points_world = (pose[:3, :3] @ points_camera.T + pose[:3, 3:4]).T

        return points_world

def track_objects_across_frames(self, current_objects, previous_objects):
    """Simple object tracking using 3D position and appearance"""
    if not previous_objects:
        return list(range(len(current_objects)))
    
    # Create cost matrix for Hungarian algorithm
    cost_matrix = np.zeros((len(current_objects), len(previous_objects)))
    
    for i, current_obj in enumerate(current_objects):
        for j, prev_obj in enumerate(previous_objects):
            # Position cost (Euclidean distance between centroids)
            pos_cost = np.linalg.norm(current_obj['centroid'] - prev_obj['centroid'])
            
            # Appearance cost (1 - cosine similarity)
            app_cost = 1 - np.dot(current_obj['clip_features'], prev_obj['clip_features'])
            
            cost_matrix[i, j] = pos_cost + 0.5 * app_cost  # Weighted combination
    
    # Simple greedy matching (replace with Hungarian algorithm for better results)
    matches = []
    used_j = set()
    
    for i in range(len(current_objects)):
        best_j = None
        best_cost = float('inf')
        
        for j in range(len(previous_objects)):
            if j not in used_j and cost_matrix[i, j] < best_cost:
                best_cost = cost_matrix[i, j]
                best_j = j
        
        if best_j is not None and best_cost < 2.0:  # Matching threshold
            matches.append((i, best_j))
            used_j.add(best_j)
        else:
            matches.append((i, -1))  # New object
    
    return matches

def compute_3d_bbox(self, points_3d):
    """Compute 3D bounding box from point cloud"""
    if len(points_3d) == 0:
        return None
    
    centroid = np.mean(points_3d, axis=0)
    bbox_min = np.min(points_3d, axis=0)
    bbox_max = np.max(points_3d, axis=0)
    bbox_size = bbox_max - bbox_min
    
    return {
        'centroid': centroid,
        'min': bbox_min,
        'max': bbox_max,
        'size': bbox_size
    }

def build_scene_graph(self):
    """Build scene graph with object relationships"""
    scene_graph = {
        'objects': [],
        'relationships': []
    }
    
    for obj in self.objects_3d:
        scene_graph['objects'].append({
            'id': obj['id'],
            'top_labels': obj['top_labels'],
            'centroid': obj['bbox_3d']['centroid'].tolist(),
            'bbox_3d': {
                'min': obj['bbox_3d']['min'].tolist(),
                'max': obj['bbox_3d']['max'].tolist(),
                'size': obj['bbox_3d']['size'].tolist()
            },
            'clip_features': obj['aggregated_features'].tolist()
        })
    
    # Add spatial relationships (simplified)
    for i, obj1 in enumerate(self.objects_3d):
        for j, obj2 in enumerate(self.objects_3d):
            if i != j:
                distance = np.linalg.norm(obj1['bbox_3d']['centroid'] - obj2['bbox_3d']['centroid'])
                if distance < 2.0:  # Proximity threshold
                    scene_graph['relationships'].append({
                        'subject': obj1['id'],
                        'object': obj2['id'],
                        'relationship': 'near',
                        'distance': float(distance)
                    })
    
    return scene_graph

def process_scene(self):
    """Main processing pipeline"""
    frame_ids = sorted([int(f.stem.split('_')[-1]) for f in (self.data_folder / 'rgb').glob('*.png')])
    previous_objects = []
    
    for frame_id in frame_ids:
        print(f"Processing frame {frame_id}")
        
        # Load frame data
        rgb, depth, pose, intrinsics = self.load_frame_data(frame_id)
        
        # Detect objects and generate masks
        masks, object_info = self.detect_objects_2d(rgb)
        
        current_objects = []
        for mask, info in zip(masks, object_info):
            # Extract CLIP features
            clip_features = self.extract_clip_features(rgb, mask)
            
            # Lift to 3D
            points_3d = self.lift_to_3d(mask, depth, pose, intrinsics)
            
            if points_3d is not None:
                # Compute 3D bounding box
                bbox_3d = self.compute_3d_bbox(points_3d)
                
                current_objects.append({
                    'frame_id': frame_id,
                    'clip_features': clip_features,
                    'points_3d': points_3d,
                    'bbox_3d': bbox_3d,
                    'category': info['category'],
                    'confidence': info['confidence']
                })
        
        # Track objects across frames
        if previous_objects:
            matches = self.track_objects_across_frames(current_objects, previous_objects)
            
            for i, (current_idx, prev_idx) in enumerate(matches):
                if prev_idx != -1:  # Matched existing object
                    obj_id = previous_objects[prev_idx]['id']
                    # Add features to existing object
                    self.objects_3d[obj_id]['clip_features_list'].append(current_objects[current_idx]['clip_features'])
                    self.objects_3d[obj_id]['points_3d_list'].append(current_objects[current_idx]['points_3d'])
                else:  # New object
                    obj_id = len(self.objects_3d)
                    self.objects_3d.append({
                        'id': obj_id,
                        'clip_features_list': [current_objects[current_idx]['clip_features']],
                        'points_3d_list': [current_objects[current_idx]['points_3d']],
                        'first_frame': frame_id
                    })
        else:
            # First frame, create new objects
            for obj in current_objects:
                obj_id = len(self.objects_3d)
                self.objects_3d.append({
                    'id': obj_id,
                    'clip_features_list': [obj['clip_features']],
                    'points_3d_list': [obj['points_3d']],
                    'first_frame': frame_id
                })
        
        previous_objects = current_objects
    
    # Post-processing: Aggregate features and compute final 3D properties
    for obj in self.objects_3d:
        # Aggregate CLIP features
        obj['aggregated_features'] = self.aggregate_features(obj['clip_features_list'])
        
        # Select top labels
        obj['top_labels'] = self.select_top_labels(obj['aggregated_features'])
        
        # Combine all 3D points and compute final bbox
        all_points = np.vstack(obj['points_3d_list'])
        obj['bbox_3d'] = self.compute_3d_bbox(all_points)
    
    # Build scene graph
    self.scene_graph = self.build_scene_graph()
    
    # Save results
    self.save_results()

def save_results(self):
    """Save scene graph and object data"""
    # Save scene graph as JSON
    with open(self.data_folder / 'scene_graph.json', 'w') as f:
        json.dump(self.scene_graph, f, indent=2)
    
    # Save object point clouds (optional)
    for obj in self.objects_3d:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.vstack(obj['points_3d_list']))
        o3d.io.write_point_cloud(str(self.data_folder / f'object_{obj["id"]}.ply'), pcd)