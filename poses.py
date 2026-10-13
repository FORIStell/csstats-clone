"""Preset OpenPose skeletons for pose-guided generation.

Keypoints use the OpenPose COCO-18 order, normalized to 0..1 (x right, y down):
0 nose, 1 neck, 2 R-shoulder, 3 R-elbow, 4 R-wrist, 5 L-shoulder, 6 L-elbow, 7 L-wrist,
8 R-hip, 9 R-knee, 10 R-ankle, 11 L-hip, 12 L-knee, 13 L-ankle, 14 R-eye, 15 L-eye, 16 R-ear, 17 L-ear.
None = joint not visible.
"""
import numpy as np
from controlnet_aux.open_pose.body import Keypoint
from controlnet_aux.open_pose.util import draw_bodypose
from PIL import Image


def _head(nx, ny, flip=False):
    """Eyes and ears around a front-facing nose (above it, or below it when upside down)."""
    d = 0.015 if flip else -0.015
    return [(nx - 0.02, ny + d), (nx + 0.02, ny + d), (nx - 0.04, ny + d / 2), (nx + 0.04, ny + d / 2)]


def _front(nose, neck, rs, re, rw, ls, le, lw, rh, rk, ra, lh, lk, la, flip=False):
    return [nose, neck, rs, re, rw, ls, le, lw, rh, rk, ra, lh, lk, la] + _head(*nose, flip=flip)


POSES = {
    "Standing": _front((0.50, 0.15), (0.50, 0.23), (0.42, 0.24), (0.39, 0.36), (0.37, 0.47), (0.58, 0.24), (0.61, 0.36),
                       (0.63, 0.47), (0.45, 0.50), (0.45, 0.68), (0.45, 0.86), (0.55, 0.50), (0.55, 0.68), (0.55, 0.86)),
    "Jumping for joy": _front((0.50, 0.24), (0.50, 0.32), (0.42, 0.33), (0.34, 0.22), (0.28, 0.11), (0.58, 0.33),
                              (0.66, 0.22), (0.72, 0.11), (0.45, 0.56), (0.38, 0.70), (0.32, 0.84), (0.55, 0.56),
                              (0.62, 0.70), (0.68, 0.84)),
    "Handstand": _front((0.50, 0.77), (0.50, 0.67), (0.42, 0.68), (0.41, 0.79), (0.40, 0.90), (0.58, 0.68), (0.59, 0.79),
                        (0.60, 0.90), (0.45, 0.42), (0.45, 0.26), (0.45, 0.10), (0.55, 0.42), (0.55, 0.26), (0.55, 0.10),
                        flip=True),
    "Cartwheel": _front((0.50, 0.76), (0.50, 0.66), (0.43, 0.67), (0.40, 0.78), (0.37, 0.90), (0.57, 0.67), (0.60, 0.78),
                        (0.63, 0.90), (0.46, 0.43), (0.33, 0.29), (0.18, 0.16), (0.54, 0.43), (0.67, 0.29), (0.82, 0.16),
                        flip=True),
    "Split": _front((0.50, 0.29), (0.50, 0.37), (0.42, 0.38), (0.32, 0.38), (0.20, 0.36), (0.58, 0.38), (0.68, 0.38),
                    (0.80, 0.36), (0.47, 0.70), (0.30, 0.74), (0.12, 0.76), (0.53, 0.70), (0.70, 0.74), (0.88, 0.76)),
    "Sitting cross-legged": _front((0.50, 0.25), (0.50, 0.33), (0.42, 0.34), (0.38, 0.46), (0.42, 0.58), (0.58, 0.34),
                                   (0.62, 0.46), (0.58, 0.58), (0.45, 0.62), (0.30, 0.72), (0.56, 0.77), (0.55, 0.62),
                                   (0.70, 0.72), (0.44, 0.77)),
    # Side views: only the near-side eye/ear is visible.
    "Lying on back": [(0.17, 0.62), (0.25, 0.65), (0.26, 0.64), (0.36, 0.67), (0.45, 0.66), (0.26, 0.66), (0.35, 0.66),
                      (0.44, 0.65), (0.52, 0.65), (0.68, 0.65), (0.84, 0.66), (0.52, 0.66), (0.68, 0.66), (0.84, 0.67),
                      (0.17, 0.60), None, (0.21, 0.61), None],
    "Backflip (tucked)": [(0.52, 0.70), (0.50, 0.62), (0.49, 0.61), (0.42, 0.52), (0.40, 0.42), (0.51, 0.62),
                          (0.44, 0.53), (0.42, 0.43), (0.52, 0.40), (0.38, 0.33), (0.46, 0.22), (0.53, 0.41),
                          (0.40, 0.34), (0.48, 0.23), (0.50, 0.72), None, (0.54, 0.68), None],
}

# Text hints that help the model read the pose correctly.
POSE_HINTS = {
    "Handstand": "doing a handstand, upside down",
    "Cartwheel": "doing a cartwheel, upside down, legs spread",
    "Backflip (tucked)": "doing a backflip in mid air, upside down, knees tucked",
    "Lying on back": "lying on their back, side view",
    "Split": "doing a split on the floor",
    "Sitting cross-legged": "sitting cross-legged on the floor",
    "Jumping for joy": "jumping in the air with arms raised",
}


def pose_image(name, size=512):
    """Render a preset pose as an OpenPose skeleton image (black background)."""
    kps = [Keypoint(x, y) if p is not None else None for p in POSES[name] for x, y in [p or (0, 0)]]
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    return Image.fromarray(draw_bodypose(canvas, kps))
