"""
Mustache and beard overlay module with alpha blending and eye-based rotation.
Handles positioning, scaling, and natural blending of facial hair accessories.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class MustacheOverlay:
    """
    Handles mustache and beard overlay on detected faces.
    Supports alpha blending, scaling, and rotation based on eye positions.
    """
    
    def __init__(self, 
                 mustache_path: str = None,
                 eye_cascade_path: str = None):
        """
        Initialize mustache overlay.
        
        Args:
            mustache_path: Path to PNG mustache image with alpha channel
            eye_cascade_path: Path to Haar cascade for eye detection
        """
        self.mustache_img = None
        self.mustache_alpha = None
        
        if mustache_path and Path(mustache_path).exists():
            self.load_mustache(mustache_path)
        
        # Eye detector for rotation
        self.eye_cascade = None
        if eye_cascade_path and Path(eye_cascade_path).exists():
            self.eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    
    def load_mustache(self, path: str):
        """
        Load mustache PNG with alpha channel.
        
        Args:
            path: Path to PNG image
        """
        # Load with alpha channel
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if img is None:
            logger.error(f"Failed to load mustache image from {path}")
            return
        
        # Check if has alpha channel
        if img.shape[2] == 4:
            self.mustache_img = img[:, :, :3]  # BGR channels
            self.mustache_alpha = img[:, :, 3] / 255.0  # Normalize alpha to 0-1
        else:
            # No alpha channel, create one
            self.mustache_img = img
            self.mustache_alpha = np.ones((img.shape[0], img.shape[1]), dtype=np.float32)
        
        logger.info(f"Loaded mustache image: {self.mustache_img.shape}")
    
    def detect_eyes(self, 
                    face_roi: np.ndarray, 
                    face_box: Tuple[int, int, int, int]) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        """
        Detect eyes in face ROI and return their absolute positions.
        
        Args:
            face_roi: Cropped face region
            face_box: (x, y, w, h) of face in original image
        
        Returns:
            Tuple of (left_eye_center, right_eye_center) or None
        """
        if self.eye_cascade is None:
            return None
        
        fx, fy, fw, fh = face_box
        
        # Convert to grayscale
        if len(face_roi.shape) == 3:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_roi
        
        # Detect eyes
        eyes = self.eye_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(int(fw * 0.1), int(fh * 0.1)),
            maxSize=(int(fw * 0.4), int(fh * 0.4))
        )
        
        if len(eyes) < 2:
            return None
        
        # Get two largest eyes
        eyes_sorted = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]
        
        # Sort by x-coordinate (left to right)
        eyes_sorted = sorted(eyes_sorted, key=lambda e: e[0])
        
        # Calculate eye centers in absolute coordinates
        left_eye = eyes_sorted[0]
        right_eye = eyes_sorted[1]
        
        left_center = (
            fx + left_eye[0] + left_eye[2] // 2,
            fy + left_eye[1] + left_eye[3] // 2
        )
        right_center = (
            fx + right_eye[0] + right_eye[2] // 2,
            fy + right_eye[1] + right_eye[3] // 2
        )
        
        return left_center, right_center
    
    def calculate_rotation_angle(self, 
                                 left_eye: Tuple[int, int], 
                                 right_eye: Tuple[int, int]) -> float:
        """
        Calculate rotation angle from eye positions.
        
        Args:
            left_eye: (x, y) of left eye center
            right_eye: (x, y) of right eye center
        
        Returns:
            Rotation angle in degrees
        """
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]
        
        angle = np.degrees(np.arctan2(dy, dx))
        return angle
    
    def rotate_image(self, 
                    image: np.ndarray, 
                    alpha: np.ndarray, 
                    angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rotate image and alpha mask.
        
        Args:
            image: Input image
            alpha: Alpha mask
            angle: Rotation angle in degrees
        
        Returns:
            Tuple of (rotated_image, rotated_alpha)
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Rotate image
        rotated_img = cv2.warpAffine(
            image, 
            M, 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0)
        )
        
        # Rotate alpha
        rotated_alpha = cv2.warpAffine(
            alpha, 
            M, 
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return rotated_img, rotated_alpha
    
    def overlay_mustache(self, 
                        frame: np.ndarray, 
                        face_box: Tuple[int, int, int, int],
                        scale_factor: float = 1.3,
                        y_offset_ratio: float = 0.55,
                        enable_rotation: bool = True) -> np.ndarray:
        """
        Overlay mustache on face.
        
        Args:
            frame: Input frame
            face_box: (x, y, w, h) of detected face
            scale_factor: Width scale relative to face width
            y_offset_ratio: Vertical position (0=top, 1=bottom of face)
            enable_rotation: Whether to apply rotation based on eyes
        
        Returns:
            Frame with mustache overlay
        """
        if self.mustache_img is None:
            return frame
        
        # Force all coordinates to integers to prevent floating point jitter
        fx, fy, fw, fh = int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])
        
        # Calculate mustache size
        mustache_width = int(fw * scale_factor)
        aspect_ratio = self.mustache_img.shape[0] / self.mustache_img.shape[1]
        mustache_height = int(mustache_width * aspect_ratio)
        
        # Resize mustache
        mustache_resized = cv2.resize(
            self.mustache_img, 
            (mustache_width, mustache_height),
            interpolation=cv2.INTER_AREA
        )
        alpha_resized = cv2.resize(
            self.mustache_alpha,
            (mustache_width, mustache_height),
            interpolation=cv2.INTER_AREA
        )
        
        # Calculate position
        mustache_x = fx + (fw - mustache_width) // 2
        mustache_y = fy + int(fh * y_offset_ratio)
        
        # Optional: Rotate based on eye positions
        rotation_applied = False
        if enable_rotation and self.eye_cascade is not None:
            face_roi = frame[fy:fy+fh, fx:fx+fw]
            eyes = self.detect_eyes(face_roi, face_box)
            
            if eyes is not None:
                left_eye, right_eye = eyes
                angle = self.calculate_rotation_angle(left_eye, right_eye)
                
                # Only rotate if angle is reasonable (-30 to 30 degrees)
                if -30 <= angle <= 30:
                    mustache_resized, alpha_resized = self.rotate_image(
                        mustache_resized, alpha_resized, angle
                    )
                    rotation_applied = True
        
        # Apply overlay with alpha blending
        result = self.alpha_blend(
            frame, 
            mustache_resized, 
            alpha_resized,
            mustache_x, 
            mustache_y
        )
        
        return result
    
    def alpha_blend(self, 
                   background: np.ndarray,
                   foreground: np.ndarray,
                   alpha: np.ndarray,
                   x: int,
                   y: int) -> np.ndarray:
        """
        Alpha blend foreground onto background at position (x, y).
        
        Args:
            background: Background image
            foreground: Foreground image
            alpha: Alpha mask (0-1)
            x, y: Position to place foreground
        
        Returns:
            Blended image
        """
        result = background.copy()
        
        h, w = foreground.shape[:2]
        bg_h, bg_w = background.shape[:2]
        
        # Clamp to image boundaries
        if x >= bg_w or y >= bg_h or x + w < 0 or y + h < 0:
            return result
        
        # Calculate valid region
        src_x1 = max(0, -x)
        src_y1 = max(0, -y)
        src_x2 = min(w, bg_w - x)
        src_y2 = min(h, bg_h - y)
        
        dst_x1 = max(0, x)
        dst_y1 = max(0, y)
        dst_x2 = dst_x1 + (src_x2 - src_x1)
        dst_y2 = dst_y1 + (src_y2 - src_y1)
        
        # Extract regions
        fg_region = foreground[src_y1:src_y2, src_x1:src_x2]
        alpha_region = alpha[src_y1:src_y2, src_x1:src_x2]
        bg_region = result[dst_y1:dst_y2, dst_x1:dst_x2]
        
        # Expand alpha to 3 channels
        if len(alpha_region.shape) == 2:
            alpha_region = np.stack([alpha_region] * 3, axis=2)
        
        # Blend
        blended = (fg_region * alpha_region + bg_region * (1 - alpha_region)).astype(np.uint8)
        
        # Place back
        result[dst_y1:dst_y2, dst_x1:dst_x2] = blended
        
        return result


class MustacheGallery:
    """
    Manager for multiple mustache styles.
    """
    
    def __init__(self, assets_dir: str = None):
        """
        Initialize gallery.
        
        Args:
            assets_dir: Directory containing mustache PNG files
        """
        self.mustaches = {}
        self.current_style = None
        
        if assets_dir:
            self.load_from_directory(assets_dir)
    
    def load_from_directory(self, directory: str):
        """Load all PNG files from directory."""
        assets_path = Path(directory)
        
        if not assets_path.exists():
            logger.warning(f"Assets directory {directory} does not exist")
            return
        
        for png_file in assets_path.glob("*.png"):
            style_name = png_file.stem
            self.mustaches[style_name] = str(png_file)
            logger.info(f"Loaded style: {style_name}")
        
        if self.mustaches and self.current_style is None:
            self.current_style = list(self.mustaches.keys())[0]
    
    def get_style(self, style_name: str) -> Optional[str]:
        """Get path to mustache style."""
        return self.mustaches.get(style_name)
    
    def get_styles(self) -> list:
        """Get list of available styles."""
        return list(self.mustaches.keys())
    
    def set_current_style(self, style_name: str):
        """Set current active style."""
        if style_name in self.mustaches:
            self.current_style = style_name
    
    def get_current_path(self) -> Optional[str]:
        """Get path to current style."""
        if self.current_style:
            return self.mustaches[self.current_style]
        return None
    
    def next_style(self):
        """Switch to next style."""
        if not self.mustaches:
            return
        
        styles = list(self.mustaches.keys())
        if self.current_style in styles:
            idx = styles.index(self.current_style)
            self.current_style = styles[(idx + 1) % len(styles)]
        else:
            self.current_style = styles[0]
    
    def prev_style(self):
        """Switch to previous style."""
        if not self.mustaches:
            return
        
        styles = list(self.mustaches.keys())
        if self.current_style in styles:
            idx = styles.index(self.current_style)
            self.current_style = styles[(idx - 1) % len(styles)]
        else:
            self.current_style = styles[0]
