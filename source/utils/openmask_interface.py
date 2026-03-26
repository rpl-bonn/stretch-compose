import json
import os
import shutil
import zipfile

import numpy as np
import torch

import clip
import open3d as o3d
import requests
from urllib3.exceptions import ReadTimeoutError
from utils import recursive_config
from utils.docker_communication import _get_content
from utils.recursive_config import Config
from sklearn.cluster import DBSCAN
import cv2

MODEL, PREPROCESS = clip.load("ViT-L/14@336px", device="cpu")

def select_with_clip(crops: list[dict], query: str, device="cpu", top_k: int = 1, VIS_BLOCK: bool = False, image_path=None, img_dir: str="/home/ws/data/images/") -> list[dict]:
    """
    Compare SAM2 crops with a query object using CLIP and return the most similar ones.

    Args:
        crops (list[dict]): list of dicts from sam_random_detect with keys:
                            "crop" (PIL.Image), "mask", "score", "logits"
        query (str): target object name to search for
        device (str): 'cpu' or 'cuda'
        top_k (int): number of best matches to return
        VIS_BLOCK (bool): if True, visualize results on original image
        image (PIL.Image or None): full RGB image (needed for visualization)

    Returns:
        list[dict]: top-k crop entries with bbox and similarity
    """
    if not crops:
        return []

    # Preprocess crops
    images = [PREPROCESS(c["crop"]).unsqueeze(0).to(device) for c in crops]
    image_batch = torch.cat(images, dim=0)

    # Encode images and text
    with torch.no_grad():
        img_features = MODEL.encode_image(image_batch)
        txt_features = MODEL.encode_text(clip.tokenize([query]).to(device))

    # Normalize
    img_features /= img_features.norm(dim=-1, keepdim=True)
    txt_features /= txt_features.norm(dim=-1, keepdim=True)

    # Cosine similarity
    sims = (img_features @ txt_features.T).squeeze(1).cpu().numpy()
    top_idx = np.argsort(sims)[::-1][:top_k]

    results = []
    for idx in top_idx:
        mask = crops[idx]["mask"]
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            continue
        bbox = (xs.min(), ys.min(), xs.max(), ys.max())
        entry = {
            "crop": crops[idx]["crop"],
            "mask": mask,
            "score": crops[idx]["score"],
            "logits": crops[idx]["logits"],
            "bbox": bbox,
            "similarity": sims[idx]
        }
        print("found bbox", bbox, f"similarity {sims[idx]:.3f}")
        results.append(entry)

    # Visualization
    if VIS_BLOCK and image_path is not None:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("The 'image' argument must be provided for visualization (VIS_BLOCK=True).")
        img_cv = np.array(image)[:, :, ::-1].copy()  # PIL to OpenCV BGR
        IMG_DIR = img_dir
        os.makedirs(IMG_DIR, exist_ok=True)
        i = 0
        for r in results:
            img_cv = np.array(image)[:, :, ::-1].copy()  # PIL to OpenCV BGR
            i += 1
            x0, y0, x1, y1 = r["bbox"]
            cv2.rectangle(img_cv, (x0, y0), (x1, y1), (0, 255, 0), 2)
            mask = r["mask"].astype(np.uint8) * 255
            mask_rgb = cv2.merge([mask, mask, mask])
            masked_img = cv2.addWeighted(img_cv, 0.7, mask_rgb, 0.3, 0)
            img_cv = masked_img
            cv2.putText(
                img_cv, f"{query}: {r['similarity']:.2f}",
                (x0, max(0, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (0, 255, 0), 1, cv2.LINE_AA
            )
        # Save visualization to IMG_DIR/clip_{idx}
            out_path = os.path.join(IMG_DIR, f"clip_{i}.png")
            print(f"Saved CLIP visualization to {out_path}")
            cv2.imwrite(out_path, img_cv)

    return results

def zip_point_cloud(path: str) -> str:
    name = os.path.basename(path)
    if os.path.exists(name):
        shutil.rmtree(name)
    output_filename = os.path.join(path, f"{name}.zip")
    with zipfile.ZipFile(output_filename, "w") as zipf:
        for foldername, subfolders, filenames in os.walk(path):
            for filename in filenames:
                if filename.endswith(".zip"):
                    continue
                file_path = os.path.join(foldername, filename)
                zipf.write(file_path, os.path.relpath(file_path, path))
    return output_filename


def get_mask_clip_features() -> None:
    # CONSTANTS
    PORT = 5001
    SAVE_PATH = "./tmp"

    config = recursive_config.Config()
    directory_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    directory_path = os.path.join(str(directory_path), ending)
    zipfile = zip_point_cloud(directory_path)

    kwargs = {
        "name": ("str", ending),
        "overwrite": ("bool", True),
        "scene_intrinsic_resolution": ("str", "[1440,1920]"),
        # "scene_intrinsic_resolution": ("str", "[968,1296]"),
    }
    server_address = f"http://localhost:{PORT}/openmask/save_and_predict"
    with open(zipfile, "rb") as f:
        try:
            response = requests.post(server_address, files={"scene": f}, params=kwargs, timeout=900)
        except ReadTimeoutError:
            print("Request timed out!")
            return

    if response.status_code == 200:  # fail
        contents = _get_content(response, SAVE_PATH)
    else:
        try:
            message = json.loads(response.content)
            print(f"{message['error']}", f"Status code: {response.status_code}", sep="\n")
            return
        except json.JSONDecodeError:
            print(f"Failed to decode JSON response: {response.content}")
            return

    features = contents["clip_features"]
    masks = contents["scene_MASKS"]

    save_path = config.get_subpath("openmask_features")
    save_path = os.path.join(save_path, ending)
    os.makedirs(save_path, exist_ok=True)
    feature_path = os.path.join(str(save_path), "clip_features.npy")
    mask_path = os.path.join(str(save_path), "scene_MASKS.npy")
    np.save(feature_path, features)
    np.save(mask_path, masks)

    # make unique
    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    features = features[mask_idx]
    feature_compressed_path = os.path.join(str(save_path), "clip_features_comp.npy")
    mask_compressed_path = os.path.join(str(save_path), "scene_MASKS_comp.npy")
    np.save(feature_compressed_path, features)
    np.save(mask_compressed_path, masks)


def get_mask_points(item: str, config, idx: int = 0, vis_block: bool = False):
    pcd_name = config["pre_scanned_graphs"]["high_res"]
    print(f"pcd path name {pcd_name}")
    base_path = config.get_subpath("openmask_features")
    feat_path = os.path.join(base_path, pcd_name, "clip_features_comp.npy")
    mask_path = os.path.join(base_path, pcd_name, "scene_MASKS_comp.npy")
    pcd_path = os.path.join(config.get_subpath("aligned_point_clouds"), pcd_name, "scene.ply")

    features = np.load(feat_path)
    masks = np.load(mask_path)
    item = item.lower()

    features, feat_idx = np.unique(features, axis=0, return_index=True)
    masks = masks[:, feat_idx]
    # masks, mask_idx = np.unique(masks, axis=1, return_index=True)
    # features = features[mask_idx]

    text = clip.tokenize([item]).to("cpu")
    #print(f"Searching for '{item}' in {features.shape[0]} features...")

    # Compute the CLIP feature vector for the specified word
    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    cos_sim = torch.nn.functional.cosine_similarity(torch.Tensor(features), text_features, dim=1)
    values, indices = torch.topk(cos_sim, idx + 1)
    most_sim_feat_idx = indices[-1].item()
    #if values[-1].item() > 0.2:
    print(f"{item}: {most_sim_feat_idx=}", f"value={values[-1].item()}")
    # idx = 1
    mask = masks[:, most_sim_feat_idx].astype(bool)

    pcd = o3d.io.read_point_cloud(str(pcd_path))

    # Apply DBSCAN clustering to the object masks & select largest cluster
    selected_points = np.where(mask)[0]
    points = np.asarray(pcd.points)
    selected_point_coords = points[selected_points]

    db = DBSCAN(eps=0.05, min_samples=10).fit(selected_point_coords)

    unique_labels, counts = np.unique(db.labels_, return_counts=True)
    largest_cluster_label = unique_labels[np.argmax(counts)]
    cluster_points_idx = selected_points[db.labels_ == largest_cluster_label]

    pcd_in = pcd.select_by_index(cluster_points_idx)
    #pcd_in = pcd.select_by_index(np.where(mask)[0])
    pcd_out = pcd.select_by_index(np.where(~mask)[0])

    if vis_block:
        pcd_in.paint_uniform_color([1, 0, 1])
        # Visualize with origin coordinate system
        origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
        o3d.visualization.draw_geometries([pcd_in, pcd_out, origin], window_name=item)

    return pcd_in, pcd_out, values[-1].item()

def get_text_similarity(object_label: str, furniture_label: str):
    # Tokenize both labels
    labels = [object_label.lower(), furniture_label.lower()]
    text = clip.tokenize(labels).to("cpu")

    # Encode text features with CLIP
    with torch.no_grad():
        text_features = MODEL.encode_text(text)

    # Compute cosine similarity between the two label features
    similarity = torch.nn.functional.cosine_similarity(
        text_features[0].unsqueeze(0), text_features[1].unsqueeze(0), dim=1
    )

    return similarity.item()

########################################################################################
# TESTING
########################################################################################



def _test_mask_points():
    vocab_file = "/home/ws/data/open_vocab.json" 
    with open(vocab_file, "r") as f:
        vocab = json.load(f)                   # list of strings
        
    # for item in vocab:
    #     print(f"\nItem: {item}")
    #     config = Config()
    #     get_mask_points(item, config, idx=0, vis_block=True)
    item = "blue water bottle"
    config = Config()
    for i in range(15):
        print(i, end=", ")
        get_mask_points(item, config, idx=i, vis_block=True)


if __name__ == "__main__":
    _test_mask_points()
