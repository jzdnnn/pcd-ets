"""
Utility functions for the mustache overlay pipeline.
Includes NMS, timing, logging, and visualization helpers.
"""

import cv2
import numpy as np
import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any
import logging


def setup_logging(log_file: str = None, level=logging.INFO):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def compute_iou(box1: Tuple[int, int, int, int], 
                box2: Tuple[int, int, int, int]) -> float:
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1, box2: (x, y, w, h) format
    
    Returns:
        IoU score between 0 and 1
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2) format
    box1_coords = [x1, y1, x1 + w1, y1 + h1]
    box2_coords = [x2, y2, x2 + w2, y2 + h2]
    
    # Compute intersection
    x_left = max(box1_coords[0], box2_coords[0])
    y_top = max(box1_coords[1], box2_coords[1])
    x_right = min(box1_coords[2], box2_coords[2])
    y_bottom = min(box1_coords[3], box2_coords[3])
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    
    # Compute union
    box1_area = w1 * h1
    box2_area = w2 * h2
    union = box1_area + box2_area - intersection
    
    return intersection / union if union > 0 else 0.0


def non_max_suppression(boxes: List[Tuple[int, int, int, int]], 
                        scores: List[float], 
                        iou_threshold: float = 0.3) -> List[int]:
    """
    Apply Non-Maximum Suppression to filter overlapping detections.
    
    Args:
        boxes: List of (x, y, w, h) bounding boxes
        scores: List of confidence scores for each box
        iou_threshold: IoU threshold for suppression
    
    Returns:
        List of indices of boxes to keep
    """
    if len(boxes) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Get indices sorted by score (descending)
    indices = np.argsort(scores)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Keep the box with highest score
        current = indices[0]
        keep.append(current)
        
        if len(indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        ious = np.array([
            compute_iou(tuple(boxes[current]), tuple(boxes[idx]))
            for idx in indices[1:]
        ])
        
        # Keep only boxes with IoU below threshold
        indices = indices[1:][ious < iou_threshold]
    
    return keep


class Timer:
    """Simple timer for performance measurement."""
    
    def __init__(self):
        self.times = {}
        self.starts = {}
    
    def start(self, name: str):
        """Start timing an operation."""
        self.starts[name] = time.time()
    
    def stop(self, name: str):
        """Stop timing and record duration."""
        if name in self.starts:
            duration = time.time() - self.starts[name]
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(duration)
            del self.starts[name]
            return duration
        return None
    
    def get_avg(self, name: str) -> float:
        """Get average time for an operation."""
        if name in self.times and len(self.times[name]) > 0:
            return sum(self.times[name]) / len(self.times[name])
        return 0.0
    
    def report(self) -> Dict[str, float]:
        """Get timing report."""
        return {name: self.get_avg(name) for name in self.times}


def draw_boxes(image: np.ndarray, 
               boxes: List[Tuple[int, int, int, int]], 
               labels: List[str] = None,
               color: Tuple[int, int, int] = (0, 255, 0),
               thickness: int = 2) -> np.ndarray:
    """
    Draw bounding boxes on image.
    
    Args:
        image: Input image
        boxes: List of (x, y, w, h) boxes
        labels: Optional labels for each box
        color: Box color in BGR
        thickness: Line thickness
    
    Returns:
        Image with drawn boxes
    """
    result = image.copy()
    
    for i, (x, y, w, h) in enumerate(boxes):
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        
        if labels and i < len(labels):
            label = labels[i]
            # Add text background
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
            )
            cv2.rectangle(
                result, 
                (x, y - text_h - 4), 
                (x + text_w, y), 
                color, 
                -1
            )
            cv2.putText(
                result, 
                label, 
                (x, y - 4), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.6, 
                (255, 255, 255), 
                1
            )
    
    return result


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to JSON file."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(config, f, indent=2, cls=NumpyEncoder)


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)


def ensure_dir(path: str):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def visualize_keypoints(image: np.ndarray, 
                        keypoints: List[cv2.KeyPoint],
                        color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
    """
    Visualize ORB keypoints on image.
    
    Args:
        image: Input image
        keypoints: List of cv2.KeyPoint objects
        color: Color for keypoints
    
    Returns:
        Image with drawn keypoints
    """
    return cv2.drawKeypoints(
        image, 
        keypoints, 
        None, 
        color=color,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )


def clamp_to_image(x: int, y: int, w: int, h: int, 
                   img_h: int, img_w: int) -> Tuple[int, int, int, int]:
    """
    Clamp bounding box coordinates to image boundaries.
    
    Args:
        x, y, w, h: Bounding box
        img_h, img_w: Image dimensions
    
    Returns:
        Clamped (x, y, w, h)
    """
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = max(1, min(w, img_w - x))
    h = max(1, min(h, img_h - y))
    return x, y, w, h


def calculate_fps(frame_times: List[float], window: int = 30) -> float:
    """
    Calculate FPS from frame timestamps.
    
    Args:
        frame_times: List of frame timestamps
        window: Window size for averaging
    
    Returns:
        Average FPS
    """
    if len(frame_times) < 2:
        return 0.0
    
    recent = frame_times[-window:]
    if len(recent) < 2:
        return 0.0
    
    time_diff = recent[-1] - recent[0]
    if time_diff == 0:
        return 0.0
    
    return (len(recent) - 1) / time_diff
