import sys
import os
from collections import OrderedDict

sys.path.append(os.path.join(os.path.dirname(__file__), "RAFT"))

import torch
import numpy as np
from tqdm import tqdm

from utils.video import extract_frames
from utils.flow import load_image, compute_flow

from RAFT.core.raft import RAFT
from RAFT.core.utils.utils import InputPadder
import argparse


def load_raft(weights, device):
    args = argparse.Namespace(
        small=False,
        mixed_precision=False,
        alternate_corr=False
    )

    model = RAFT(args)
    # Load weights to CPU first to avoid CUDA deserialization error
    state_dict = torch.load(weights, map_location=torch.device('cpu'))
    
    # Handle DataParallel weights (remove 'module.' prefix)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
        
    model.load_state_dict(new_state_dict)
    model.to(device)
    model.eval()
    return model


def main():
    video_path = "data/videos/input.mov"
    frame_dir = "data/frames"
    flow_dir = "flows"
    os.makedirs(flow_dir, exist_ok=True)

    extract_frames(video_path, frame_dir)

    # Check for CUDA, then MPS (Apple Silicon), then CPU
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")

    model = load_raft("RAFT/models/raft-things.pth", device)

    frames = sorted(os.listdir(frame_dir))

    for i in tqdm(range(len(frames) - 1)):
        f1 = load_image(f"{frame_dir}/{frames[i]}")
        f2 = load_image(f"{frame_dir}/{frames[i+1]}")

        padder = InputPadder(f1.shape)
        f1, f2 = padder.pad(f1, f2)

        flow_fwd = compute_flow(model, f1, f2, device)
        flow_bwd = compute_flow(model, f2, f1, device)

        np.save(f"{flow_dir}/F_{i:05d}.npy", flow_fwd)
        np.save(f"{flow_dir}/B_{i:05d}.npy", flow_bwd)

    print("[DONE] Optical flow computed")


if __name__ == "__main__":
    main()
