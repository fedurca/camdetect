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

# Canonical names for the open-vocabulary (YOLO-World) classes that are not in
# COCO. Several prompt phrasings map to one canonical class.
OPEN_VOCAB_ALIASES = {
    "trash bin": "trash bin",
    "garbage can": "trash bin",
    "dustbin": "trash bin",
    "kick scooter": "scooter",
    "scooter": "scooter",
    "roller skates": "skates",
    "inline skates": "skates",
    "rollerblades": "skates",
    "drone": "drone",
    "quadcopter": "drone",
}


def canonical_name(name: str) -> str:
    """Normalize a detector/prompt label to a canonical class name."""
    return OPEN_VOCAB_ALIASES.get(name.strip().lower(), name)


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
    # open-vocabulary additions
    "trash bin": (74, 222, 128),   # green
    "scooter": (34, 211, 238),     # cyan
    "skates": (232, 121, 249),     # fuchsia
    "drone": (250, 250, 250),      # white
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
    "trash bin": 1.1,
    "scooter": 1.1,
    "skates": 1.7,
    "drone": 0.3,
}

# Czech display labels for the UI legend/overlays.
CLASS_LABELS_CS = {
    "person": "clovek",
    "bicycle": "kolo",
    "car": "auto",
    "motorcycle": "motorka",
    "bus": "autobus",
    "truck": "nakladni auto",
    "bird": "ptak",
    "cat": "kocka",
    "dog": "pes",
    "horse": "kun",
    "sheep": "ovce",
    "cow": "krava",
    "trash bin": "popelnice",
    "scooter": "kolobezka",
    "skates": "brusle",
    "drone": "dron",
}


def color_for(name: str) -> tuple[int, int, int]:
    return CLASS_COLORS.get(name, DEFAULT_COLOR)


def height_for(name: str, default: float = 1.7) -> float:
    return CLASS_HEIGHTS.get(name, default)


def label_cs(name: str) -> str:
    return CLASS_LABELS_CS.get(name, name)
