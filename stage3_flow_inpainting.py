import os
import numpy as np
import cv2
from utils.inpaint import inpaint_flow

FLOW_DIR = "flows"
MASK_DIR = "masks"
OUT_DIR = "flows_inpainted"

os.makedirs(OUT_DIR, exist_ok=True)

flow_files = sorted([f for f in os.listdir(FLOW_DIR) if f.startswith("F_")])

for f in flow_files:
    idx = f.split("_")[1].split(".")[0]

    flow = np.load(f"{FLOW_DIR}/F_{idx}.npy")

    mask_path = f"{MASK_DIR}/M_{idx}.npy"
    if not os.path.exists(mask_path):
        continue

    mask = np.load(mask_path)

    flow_inpainted = inpaint_flow(flow, mask)

    np.save(f"{OUT_DIR}/F_{idx}.npy", flow_inpainted)

    # visualization
    mag = np.linalg.norm(flow_inpainted, axis=2)
    mag = (mag / (mag.max() + 1e-6) * 255).astype(np.uint8)
    cv2.imwrite(f"{OUT_DIR}/F_{idx}_mag.png", mag)

print("[DONE] Flow inpainting completed")
