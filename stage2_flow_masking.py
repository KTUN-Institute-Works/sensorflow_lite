import os
import numpy as np
import cv2
from utils.masking import combined_mask

FLOW_DIR = "flows"
OUT_DIR = "masks"
WINDOW = 5   # ω = 5 (küçük başlayalım)

os.makedirs(OUT_DIR, exist_ok=True)

flow_files = sorted([f for f in os.listdir(FLOW_DIR) if f.startswith("F_")])

for i in range(WINDOW, len(flow_files) - WINDOW):
    F_list = []
    B_list = []

    for k in range(i - WINDOW, i + WINDOW):
        F_list.append(np.load(f"{FLOW_DIR}/F_{k:05d}.npy"))
        B_list.append(np.load(f"{FLOW_DIR}/B_{k:05d}.npy"))

    mask = combined_mask(F_list, B_list)

    np.save(f"{OUT_DIR}/M_{i:05d}.npy", mask)

    # visualization
    vis = (mask * 255).astype(np.uint8)
    cv2.imwrite(f"{OUT_DIR}/M_{i:05d}.png", vis)

print("[DONE] Flow masks generated")
