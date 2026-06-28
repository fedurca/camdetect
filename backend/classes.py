"""Shared COCO class names and a stable class->color map.

The same mapping is mirrored in ``frontend/colors.js`` so a given class is the
same color in the video overlays and in the 3D scene.
"""
from __future__ import annotations

# COCO id -> human label (only the subset we care about is filled; others fall
# back to ultralytics' own names at runtime).
COCO_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
    14: "bird",
    15: "cat",
    16: "dog",
    17: "horse",
    18: "sheep",
    19: "cow",
    20: "elephant",
    21: "bear",
    22: "zebra",
    23: "giraffe",
}

# class name -> RGB (0-255). Kept in sync with frontend/colors.js.
CLASS_COLORS = {
    "person": (239, 68, 68),     # red
    "bicycle": (245, 158, 11),   # amber
    "car": (59, 130, 246),       # blue
    "motorcycle": (168, 85, 247),  # purple
    "bus": (16, 185, 129),       # emerald
    "truck": (14, 165, 233),     # sky
    "bird": (250, 204, 21),      # yellow
    "cat": (244, 114, 182),      # pink
    "dog": (251, 146, 60),       # orange
    "horse": (132, 204, 22),     # lime
    "sheep": (148, 163, 184),    # slate
    "cow": (217, 119, 6),        # brown
    "elephant": (100, 116, 139),
    "bear": (120, 53, 15),
    "zebra": (226, 232, 240),
    "giraffe": (202, 138, 4),
}

DEFAULT_COLOR = (148, 163, 184)  # slate-400

# Approximate physical heights (m) per class, used for 3D box extents.
CLASS_HEIGHTS = {
    "person": 1.7,
    "bicycle": 1.1,
    "car": 1.5,
    "motorcycle": 1.3,
    "bus": 3.2,
    "truck": 3.5,
    "bird": 0.3,
    "cat": 0.3,
    "dog": 0.5,
    "horse": 1.6,
    "sheep": 1.0,
    "cow": 1.5,
}


def color_for(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


def height_for(name: str, default: float = 1.7) -> float:
    return CLASS_HEIGHTS.get(name, default)
