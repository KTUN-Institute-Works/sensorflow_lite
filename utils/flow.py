import torch
import numpy as np

def load_image(imfile):
    import cv2
    img = cv2.imread(imfile)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None] / 255.0


@torch.no_grad()
def compute_flow(model, img1, img2, device):
    img1 = img1.to(device)
    img2 = img2.to(device)

    _, flow = model(img1, img2, iters=20, test_mode=True)
    return flow[0].permute(1, 2, 0).cpu().numpy()
