import numpy as np

def inpaint_flow(flow, mask, iterations=200):
    """
    flow: HxWx2
    mask: HxW (1 = valid, 0 = invalid)
    """
    h, w, _ = flow.shape
    flow_filled = flow.copy()

    for _ in range(iterations):
        for y in range(1, h - 1):
            for x in range(1, w - 1):
                if mask[y, x] == 0:
                    neighbors = []
                    for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                        if mask[y + dy, x + dx] == 1:
                            neighbors.append(flow_filled[y + dy, x + dx])

                    if neighbors:
                        flow_filled[y, x] = np.mean(neighbors, axis=0)

    return flow_filled
