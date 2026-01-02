import os
import cv2
import numpy as np

STAB_DIR = "stabilized_frames"
VIDEO_IN = "data/videos/input.mp4"
VIDEO_OUT = "output_stabilized.mp4"
SIDE_BY_SIDE = "comparison.mp4"

frames = sorted([f for f in os.listdir(STAB_DIR) if f.endswith(".png")])

# Load stabilized frames
stab_frames = [cv2.imread(os.path.join(STAB_DIR, f)) for f in frames]

h, w = stab_frames[0].shape[:2]

# Define safe crop (10% margin – conservative)
crop_x = int(0.05 * w)
crop_y = int(0.05 * h)

crop_w = w - 2 * crop_x
crop_h = h - 2 * crop_y

# Crop & resize
stab_processed = []
for f in stab_frames:
    crop = f[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
    resized = cv2.resize(crop, (w, h))
    stab_processed.append(resized)

# Load original video
cap = cv2.VideoCapture(VIDEO_IN)
orig_frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    orig_frames.append(frame)
cap.release()

# Align frame counts
n = min(len(orig_frames), len(stab_processed))
orig_frames = orig_frames[:n]
stab_processed = stab_processed[:n]

fps = 30

# Write stabilized video
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(VIDEO_OUT, fourcc, fps, (w, h))
for f in stab_processed:
    out.write(f)
out.release()

# Side-by-side comparison
out_cmp = cv2.VideoWriter(SIDE_BY_SIDE, fourcc, fps, (w*2, h))
for o, s in zip(orig_frames, stab_processed):
    combo = np.hstack([o, s])
    out_cmp.write(combo)
out_cmp.release()

print("[DONE] Videos exported:")
print(" -", VIDEO_OUT)
print(" -", SIDE_BY_SIDE)
