import pickle


path = "/home/ws/data/ipad_scans/2026_03_18/detections.pkl"
with open(path, "rb") as f:
    data = pickle.load(f)

print(data)