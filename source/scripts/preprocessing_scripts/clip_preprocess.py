import numpy as np
import json
# === File paths ===
obj_feat_file = "/home/ws/data/openmask_features/2025_09_13/clip_features.npy"        # object embeddings (instances)
cat_feat_file = "/home/ws/data/openmask_features/2025_09_13/clip_features_comp.npy"   # category embeddings
vocab_file = "/home/ws/data/open_vocab.json"        # JSON list of category names

# === Load data ===
obj_feats = np.load(obj_feat_file)         # shape: (N_objects, D)
cat_feats = np.load(cat_feat_file)         # shape: (N_categories, D)

with open(vocab_file, "r") as f:
    vocab = json.load(f)                   # list of strings

# === Normalize (cosine similarity) ===
obj_feats = obj_feats / np.linalg.norm(obj_feats, axis=1, keepdims=True)
cat_feats = cat_feats / np.linalg.norm(cat_feats, axis=1, keepdims=True)

# === Similarity computation ===
sims = obj_feats @ cat_feats.T  # shape (N_objects, N_categories)

# === Get top-3 per object ===
topk = 3
top_ids = np.argsort(-sims, axis=1)[:, :topk]  # sort descending, take topk
top_scores = np.take_along_axis(sims, top_ids, axis=1)

# === Print results ===
for i in range(len(obj_feats)):
    print(f"\nObject {i}:")
    for rank, (idx, score) in enumerate(zip(top_ids[i], top_scores[i]), start=1):
        print(f"  Top {rank}: {vocab[idx]}  (score={score:.3f})")
