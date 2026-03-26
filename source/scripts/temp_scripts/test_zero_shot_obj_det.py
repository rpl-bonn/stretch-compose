from utils.zero_shot_object_detection import *
import time

def main():
  camera = "phone"
  for _ in range(1):
    start_time = time.time()
    detected, dict = yolo_detect_object("cracker box", camera, save_block=True, use_gemini=False)
    if detected:
        _, _, _ = sam_detect_object(camera, 0, 0, 0, input_box=dict)
    
    print(f"Total time: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()