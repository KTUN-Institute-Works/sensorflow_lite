import cv2
import os

def extract_frames(video_path, out_dir, resize=(320, 240)):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, resize)
        cv2.imwrite(f"{out_dir}/{idx:05d}.png", frame)
        idx += 1

    cap.release()
    print(f"[INFO] Extracted {idx} frames")
