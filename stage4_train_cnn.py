import os
import torch
import numpy as np
from models.stabilization_net import StabilizationNet

FLOW_DIR = "flows_inpainted"
WINDOW = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = StabilizationNet().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

flow_files = sorted([f for f in os.listdir(FLOW_DIR) if f.startswith("F_")])
indices = [int(f.split("_")[1].split(".")[0]) for f in flow_files]

def load_window(center_idx):
    flows = []
    for k in range(center_idx - WINDOW, center_idx + WINDOW + 1):
        path = f"{FLOW_DIR}/F_{k:05d}.npy"
        if not os.path.exists(path):
            return None
        flow = np.load(path)
        flows.append(flow)

    flows = np.stack(flows)  # T x H x W x 2
    flows = torch.from_numpy(flows).permute(3,0,1,2).unsqueeze(0)
    return flows.float().to(DEVICE)


for epoch in range(10):
    total = 0
    count = 0

    for idx in indices:
        flow = load_window(idx)
        if flow is None:
            continue

        pred = model(flow)

        loss = (
            torch.mean(torch.abs(pred - flow)) +
            0.1 * torch.mean(torch.abs(pred[:,:,1:] - pred[:,:,:-1])) +
            0.1 * (
                torch.mean(torch.abs(pred[:,:,:,:,1:] - pred[:,:,:,:,:-1])) +
                torch.mean(torch.abs(pred[:,:,:,1:,:] - pred[:,:,:,:-1,:]))
            )
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        count += 1

    print(f"Epoch {epoch}: loss = {total / max(count,1):.4f}")

torch.save(model.state_dict(), "stabilization_net.pth")
print("[DONE] CNN trained")
