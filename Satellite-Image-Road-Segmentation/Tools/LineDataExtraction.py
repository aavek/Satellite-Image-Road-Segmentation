"""
# Reference:
# Convert Road keypoints obtained from road mask to orientation angle mask.
# https://github.com/anilbatra2185/road_connectivity
"""

import math
import numpy as np

def getVectorMapsAngles(shape, keypoints, theta=5, bin_size=10):

    im_h, im_w = shape
    vecmap = np.zeros((im_h, im_w, 2), dtype=np.float32)
    vecmap_angles = np.zeros((im_h, im_w), dtype=np.float32)
    vecmap_angles.fill(360)
    height, width, channel = vecmap.shape
    for j in range(len(keypoints)):
        for i in range(1, len(keypoints[j])):
            a = keypoints[j][i - 1]
            b = keypoints[j][i]
            ax, ay = a[0], a[1]
            bx, by = b[0], b[1]
            bax = bx - ax
            bay = by - ay
            norm = math.sqrt(1.0 * bax * bax + bay * bay) + 1e-9
            bax /= norm
            bay /= norm

            min_w = max(int(round(min(ax, bx) - theta)), 0)
            max_w = min(int(round(max(ax, bx) + theta)), width)
            min_h = max(int(round(min(ay, by) - theta)), 0)
            max_h = min(int(round(max(ay, by) + theta)), height)

            for h in range(min_h, max_h):
                for w in range(min_w, max_w):
                    px = w - ax
                    py = h - ay
                    dis = abs(bax * py - bay * px)
                    if dis <= theta:
                        vecmap[h, w, 0] = bax
                        vecmap[h, w, 1] = bay
                        _theta = math.degrees(math.atan2(bay, bax))
                        vecmap_angles[h, w] = (_theta + 360) % 360

    vecmap_angles = (vecmap_angles / bin_size).astype(int)
    return vecmap, vecmap_angles