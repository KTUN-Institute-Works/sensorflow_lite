import os
import cv2
import numpy as np
import torch
from models.stabilization_net import StabilizationNet

VIDEO_PATH = "data/videos/input.mov"
FLOW_DIR = "flows_inpainted"
OUT_DIR = "stabilized_frames"
WINDOW = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUT_DIR, exist_ok=True)

# Load video
cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

H, W = frames[0].shape[:2]

# Load model
model = StabilizationNet().to(DEVICE)
model.load_state_dict(torch.load("stabilization_net.pth", map_location=DEVICE))
model.eval()

def load_flow_window(center_idx):
    flows = []
    for k in range(center_idx - WINDOW, center_idx + WINDOW + 1):
        path = f"{FLOW_DIR}/F_{k:05d}.npy"
        if not os.path.exists(path):
            return None
        flows.append(np.load(path))
    flows = np.stack(flows)
    flows = torch.from_numpy(flows).permute(3,0,1,2).unsqueeze(0)
    return flows.float().to(DEVICE)

for i in range(WINDOW, min(len(frames)-WINDOW, len(os.listdir(FLOW_DIR)))):
    flow_window = load_flow_window(i)
    if flow_window is None:
        continue

    with torch.no_grad():
        pred = model(flow_window)

    # Take middle frame warp
    warp = pred[0, :, WINDOW].permute(1,2,0).cpu().numpy()

    # --- RESOLUTION ALIGNMENT FIX ---

    h_flow, w_flow = warp.shape[:2]
    h_frame, w_frame = H, W

    scale_x = w_frame / w_flow
    scale_y = h_frame / h_flow

    warp_resized = cv2.resize(
        warp,
        (w_frame, h_frame),
        interpolation=cv2.INTER_LINEAR
    )

    warp_resized[..., 0] *= scale_x
    warp_resized[..., 1] *= scale_y

    grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
    map_x = (grid_x + warp_resized[..., 0]).astype(np.float32)
    map_y = (grid_y + warp_resized[..., 1]).astype(np.float32)

    stabilized = cv2.remap(
        frames[i],
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )

    cv2.imwrite(f"{OUT_DIR}/frame_{i:05d}.png", stabilized)

print("[DONE] Stabilized frames generated")
