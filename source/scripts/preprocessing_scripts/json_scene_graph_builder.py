import numpy as np
import json
from pathlib import Path
import uuid
from datetime import datetime
import clip
import torch
from sklearn.cluster import DBSCAN
import matplotlib.colors as mcolors

class JSONObjectStorageSystem:
    def __init__(self, graph_path, scene_path, data_dir="object_data"):
        # Load existing scene data
        with open(graph_path, 'r') as f:
            self.graph_data = json.load(f)
        with open(scene_path, 'r') as f:
            self.scene_data = json.load(f)
        
        # Initialize data directory
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize CLIP for feature extraction
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        
        # Load or initialize object database
        self.objects_file = self.data_dir / "objects.json"
        self.features_dir = self.data_dir / "features"
        self.features_dir.mkdir(exist_ok=True)
        
        self.objects = self._load_objects()
        self.furniture = self._extract_furniture_data()
        
        # Predefined color tags for common objects
        self.color_tags = {
            'red': ['apple', 'tomato', 'strawberry', 'fire extinguisher', 'stop sign'],
            'blue': ['water bottle', 'book', 'pen', 'mug', 'plate'],
            'green': ['plant', 'bottle', 'book', 'apple', 'cucumber'],
            'yellow': ['banana', 'lemon', 'pen', 'book', 'cup'],
            'white': ['paper', 'cup', 'plate', 'bowl', 'mouse'],
            'black': ['remote', 'phone', 'mouse', 'book', 'pen'],
            'brown': ['wood', 'table', 'chair', 'book', 'box'],
            'gray': ['laptop', 'mouse', 'remote', 'tool', 'device']
        }
        
        # Create reverse mapping for color lookup
        self.object_to_colors = {}
        for color, objects in self.color_tags.items():
            for obj in objects:
                if obj not in self.object_to_colors:
                    self.object_to_colors[obj] = []
                self.object_to_colors[obj].append(color)
    
    def _load_objects(self):
        """Load existing objects from JSON file"""
        if self.objects_file.exists():
            with open(self.objects_file, 'r') as f:
                return json.load(f)
        return {"objects": {}, "next_id": 1, "version": "1.0"}
    
    def _save_objects(self):
        """Save objects to JSON file"""
        with open(self.objects_file, 'w') as f:
            json.dump(self.objects, f, indent=2)
    
    def _extract_furniture_data(self):
        """Extract furniture data for spatial reasoning"""
        furniture = {}
        for fid, fdata in self.scene_data['furniture'].items():
            furniture[int(fid)] = {
                'label': fdata['label'],
                'centroid': np.array(fdata['centroid']),
                'dimensions': np.array(fdata['dimensions'])
            }
        return furniture
    
    def _save_features(self, object_id, features):
        """Save CLIP features to a separate file"""
        feature_file = self.features_dir / f"{object_id}.npy"
        np.save(feature_file, features)
    
    def _load_features(self, object_id):
        """Load CLIP features from file"""
        feature_file = self.features_dir / f"{object_id}.npy"
        if feature_file.exists():
            return np.load(feature_file)
        return None
    
    def extract_clip_features(self, image):
        """Extract CLIP features from an image"""
        image_preprocessed = self.clip_preprocess(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_preprocessed)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()[0]
    
    def get_top_labels(self, features, top_k=3):
        """Get top k labels for features using CLIP"""
        # Use a comprehensive list of household object categories
        object_categories = [
            "cup", "bottle", "book", "phone", "laptop", "remote", "pen", 
            "paper", "key", "wallet", "glass", "plate", "bowl", "utensil",
            "apple", "banana", "orange", "plant", "clock", "vase", "toy",
            "tool", "device", "box", "bag", "cloth", "shoe", "hat", "glasses",
            "decoration", "instrument", "electronic", "food", "drink", "container"
        ]
        
        text_inputs = torch.cat([clip.tokenize(f"a photo of a {cat}") for cat in object_categories]).to(self.device)
        
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # Calculate similarities
        similarities = (text_features @ torch.tensor(features).to(self.device).float()).cpu().numpy()
        
        # Get top-k labels
        top_indices = np.argsort(similarities)[::-1][:top_k]
        return [(object_categories[i], float(similarities[i])) for i in top_indices]
    
    def get_color_tag(self, labels):
        """Get color tag based on object labels"""
        for label, confidence in labels:
            if label in self.object_to_colors:
                colors = self.object_to_colors[label]
                return colors[0]  # Return the first associated color
        return "unknown"
    
    def find_supporting_furniture(self, centroid):
        """Find which furniture object is supporting this object"""
        best_match_id = None
        min_distance = float('inf')
        
        for fid, furniture in self.furniture.items():
            f_centroid = furniture['centroid']
            dimensions = furniture['dimensions']
            
            # Check if object is on this furniture (simple height-based check)
            vertical_distance = abs(centroid[2] - (f_centroid[2] + dimensions[2]/2))
            
            # Check if object is within furniture's XY bounds
            within_x = abs(centroid[0] - f_centroid[0]) < dimensions[0]/2 + 0.1
            within_y = abs(centroid[1] - f_centroid[1]) < dimensions[1]/2 + 0.1
            
            if within_x and within_y and vertical_distance < 0.2:
                distance = np.linalg.norm(centroid - f_centroid)
                if distance < min_distance:
                    min_distance = distance
                    best_match_id = fid
        
        return best_match_id
    
    def register_new_observation(self, frame_id, centroid, image, mask=None):
        """Register a new object observation with robust ID handling"""
        # Extract features
        features = self.extract_clip_features(image)
        
        # Get top 3 labels
        top_labels = self.get_top_labels(features, top_k=3)
        
        # Get color tag
        color_tag = self.get_color_tag(top_labels)
        
        # Find supporting furniture
        supporting_furniture = self.find_supporting_furniture(centroid)
        
        # Try to match with existing objects
        matched_object_id = self._match_with_existing_objects(features, centroid, supporting_furniture)
        
        if matched_object_id is not None:
            # Update existing object
            self._update_existing_object(matched_object_id, frame_id, centroid, features, 
                                       top_labels, color_tag, supporting_furniture)
            return matched_object_id
        else:
            # Create new object
            new_object_id = self._create_new_object(frame_id, centroid, features, 
                                                  top_labels, color_tag, supporting_furniture)
            return new_object_id
    
    def _match_with_existing_objects(self, features, centroid, supporting_furniture):
        """Match observation with existing objects using multiple strategies"""
        best_match_id = None
        best_similarity = 0.7  # Minimum similarity threshold
        
        # Strategy 1: Check objects that were recently in the same location
        if supporting_furniture is not None:
            for obj_id, obj_data in self.objects['objects'].items():
                if not obj_data['is_active']:
                    continue
                
                # Check if object was recently on this furniture
                last_location = obj_data.get('location_history', [])[-1] if obj_data.get('location_history') else None
                if last_location and last_location['supporting_furniture'] == supporting_furniture:
                    # Load features and compare
                    stored_features = self._load_features(obj_id)
                    if stored_features is not None:
                        similarity = np.dot(features, stored_features)
                        if similarity > best_similarity:
                            best_similarity = similarity
                            best_match_id = obj_id
        
        # Strategy 2: Global feature matching if location-based fails
        if best_match_id is None:
            for obj_id, obj_data in self.objects['objects'].items():
                if not obj_data['is_active']:
                    continue
                
                stored_features = self._load_features(obj_id)
                if stored_features is not None:
                    similarity = np.dot(features, stored_features)
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match_id = obj_id
        
        return best_match_id if best_similarity > 0.7 else None
    
    def _create_new_object(self, frame_id, centroid, features, top_labels, color_tag, supporting_furniture):
        """Create a new object entry"""
        object_id = str(self.objects['next_id'])
        self.objects['next_id'] += 1
        
        # Create object data
        new_object = {
            'id': object_id,
            'created_at': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            'first_seen_frame': frame_id,
            'last_seen_frame': frame_id,
            'observation_count': 1,
            'top_labels': top_labels,
            'color_tag': color_tag,
            'is_active': True,
            'is_movable': True,
            'location_history': [{
                'frame_id': frame_id,
                'centroid': centroid.tolist(),
                'supporting_furniture': supporting_furniture
            }],
            'feature_history': [frame_id]
        }
        
        # Add to objects
        self.objects['objects'][object_id] = new_object
        
        # Save features
        self._save_features(object_id, features)
        
        # Save objects
        self._save_objects()
        
        return object_id
    
    def _update_existing_object(self, object_id, frame_id, centroid, features, top_labels, color_tag, supporting_furniture):
        """Update an existing object with new observation"""
        obj_data = self.objects['objects'][object_id]
        
        # Update object data
        obj_data['last_updated'] = datetime.now().isoformat()
        obj_data['last_seen_frame'] = frame_id
        obj_data['observation_count'] += 1
        
        # Update labels if confidence is higher
        current_labels = obj_data.get('top_labels', [])
        new_confidence = top_labels[0][1] if top_labels else 0
        if new_confidence > (current_labels[0][1] if current_labels else 0):
            obj_data['top_labels'] = top_labels
            obj_data['color_tag'] = color_tag
        
        # Add to location history
        obj_data['location_history'].append({
            'frame_id': frame_id,
            'centroid': centroid.tolist(),
            'supporting_furniture': supporting_furniture
        })
        
        # Keep only recent history (last 50 observations)
        if len(obj_data['location_history']) > 50:
            obj_data['location_history'] = obj_data['location_history'][-50:]
        
        # Add to feature history
        obj_data['feature_history'].append(frame_id)
        
        # Update features (weighted average)
        old_features = self._load_features(object_id)
        if old_features is not None:
            # Weighted average: newer features have more weight
            new_features = 0.7 * features + 0.3 * old_features
            new_features /= np.linalg.norm(new_features)  # Normalize
            self._save_features(object_id, new_features)
        else:
            self._save_features(object_id, features)
        
        # Save objects
        self._save_objects()
    
    def find_object(self, query_text=None, query_image=None, location_hint=None):
        """Find objects based on text or image query"""
        results = []
        
        if query_text:
            # Text-based search
            text_input = clip.tokenize([f"a photo of a {query_text}"]).to(self.device)
            with torch.no_grad():
                text_features = self.clip_model.encode_text(text_input)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                query_features = text_features.cpu().numpy()[0]
        elif query_image:
            # Image-based search
            query_features = self.extract_clip_features(query_image)
        else:
            return results
        
        # Search through all objects
        for obj_id, obj_data in self.objects['objects'].items():
            if not obj_data['is_active']:
                continue
            
            # Load features
            obj_features = self._load_features(obj_id)
            if obj_features is not None:
                similarity = np.dot(query_features, obj_features)
                
                # Apply location filter if provided
                if location_hint:
                    last_location = obj_data['location_history'][-1]
                    location_similarity = self._check_location_similarity(last_location, location_hint)
                    similarity *= location_similarity
                
                if similarity > 0.5:  # Similarity threshold
                    results.append({
                        'object_id': obj_id,
                        'similarity': float(similarity),
                        'labels': obj_data['top_labels'],
                        'color_tag': obj_data['color_tag'],
                        'last_location': obj_data['location_history'][-1] if obj_data['location_history'] else None
                    })
        
        # Sort by similarity
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results
    
    def _check_location_similarity(self, object_location, location_hint):
        """Check how similar object location is to hint location"""
        # Simple implementation: 1.0 if same furniture, 0.5 if nearby, 0.1 otherwise
        if object_location['supporting_furniture'] == location_hint:
            return 1.0
        
        # Check if furniture is nearby
        obj_furniture = object_location['supporting_furniture']
        hint_furniture = location_hint
        
        if obj_furniture in self.furniture and hint_furniture in self.furniture:
            obj_centroid = self.furniture[obj_furniture]['centroid']
            hint_centroid = self.furniture[hint_furniture]['centroid']
            distance = np.linalg.norm(obj_centroid - hint_centroid)
            
            if distance < 2.0:  # Within 2 meters
                return 0.8 - (distance / 5.0)  # Scale with distance
        
        return 0.1
    
    def get_object_history(self, object_id):
        """Get complete history of an object"""
        if object_id in self.objects['objects']:
            return self.objects['objects'][object_id]
        return None
    
    def export_for_visualization(self):
        """Export data for visualization"""
        visualization_data = {
            'objects': {},
            'furniture': self.furniture,
            'timestamp': datetime.now().isoformat()
        }
        
        for obj_id, obj_data in self.objects['objects'].items():
            if obj_data['is_active']:
                last_location = obj_data['location_history'][-1] if obj_data['location_history'] else None
                visualization_data['objects'][obj_id] = {
                    'labels': obj_data['top_labels'],
                    'color_tag': obj_data['color_tag'],
                    'current_location': last_location,
                    'observation_count': obj_data['observation_count']
                }
        
        return visualization_data

# Usage example
if __name__ == "__main__":
    # Initialize the system
    object_system = JSONObjectStorageSystem("graph.json", "scene.json", "object_data")
    
    # Example: Process a new frame
    # (In practice, you'd load actual image data)
    example_centroid = np.array([-2.5, 0.3, 0.6])  # Example 3D position
    example_image = None  # Would be actual image data in practice
    
    # Register a new observation
    object_id = object_system.register_new_observation(
        frame_id=100, 
        centroid=example_centroid, 
        image=example_image
    )
    
    print(f"Registered/updated object with ID: {object_id}")
    
    # Search for objects
    search_results = object_system.find_object(query_text="cup")
    print("Search results for 'cup':")
    for result in search_results[:3]:  # Top 3 results
        print(f"Object {result['object_id']}: {result['labels'][0][0]} ({result['similarity']:.2f})")
    
    # Export for visualization
    viz_data = object_system.export_for_visualization()
    with open("visualization_data.json", "w") as f:
        json.dump(viz_data, f, indent=2)