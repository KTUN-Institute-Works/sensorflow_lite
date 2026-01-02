import numpy as np
import cv2

def warp_flow(flow, ref_flow):
    """
    flow: HxWx2
    ref_flow: HxWx2
    """
    h, w = flow.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + ref_flow[..., 0]).astype(np.float32)
    map_y = (grid_y + ref_flow[..., 1]).astype(np.float32)

    warped = cv2.remap(
        flow, map_x, map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )
    return warped


def cycle_consistency_mask(F_list, B_list, threshold=1.0):
    """
    Implements Eq (6) from paper
    F_list, B_list: list of flows around t
    """
    h, w, _ = F_list[0].shape
    mask = np.ones((h, w), dtype=np.uint8)

    for F, B in zip(F_list, B_list):
        F_hat = warp_flow(F, F)
        B_hat = warp_flow(B, B)

        err_fwd = np.linalg.norm(F_hat + B, axis=2)
        err_bwd = np.linalg.norm(B_hat + F, axis=2)

        mask &= (err_fwd < threshold).astype(np.uint8)
        mask &= (err_bwd < threshold).astype(np.uint8)

    return mask

def fundamental_mask(flow, threshold=1.0):
    """
    flow: HxWx2  (t -> t+1)
    """
    h, w = flow.shape[:2]

    ys, xs = np.mgrid[0:h, 0:w]
    pts1 = np.stack([xs, ys], axis=-1).reshape(-1, 2)
    pts2 = pts1 + flow.reshape(-1, 2)

    pts1 = pts1.astype(np.float32)
    pts2 = pts2.astype(np.float32)

    F, mask = cv2.findFundamentalMat(
        pts1, pts2,
        method=cv2.FM_RANSAC,
        ransacReprojThreshold=threshold,
        confidence=0.99
    )

    if mask is None:
        return np.zeros((h, w), dtype=np.uint8)

    return mask.reshape(h, w).astype(np.uint8)

def combined_mask(F_list, B_list):
    mc = cycle_consistency_mask(F_list, B_list)
    mf = fundamental_mask(F_list[len(F_list)//2])
    return mc & mf
